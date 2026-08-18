"""Packet-edit environment and target-model registry."""

import bisect
from copy import deepcopy

import torch

from advGenerate.environments.NetBeacon import NetBeaconEvaluator
from advGenerate.rl_dataset import RLFlowDataset
from advGenerate.rl_utils import clone_state, generate_mask
from trafficMimic.dataset.vocab import SizeVocab, TimeVocab
from trafficMimic.model.architecture import BERT, BERTLM
from trafficMimic.utils import resolve_repo_path, set_device


class BaseEnv:
    """A tokenised flow plus physical packet features changed by each action."""

    def __init__(self, rl_args, bert_args, mode="train"):
        self.rl_args = rl_args
        self.device = set_device(rl_args.trainer.device)
        self.state_dim = int(rl_args.model.state_dim)
        if self.state_dim != int(bert_args.model.seq_len):
            raise ValueError("RL state_dim must equal traffic-BERT seq_len")

        self.timevocab = TimeVocab.load_vocab(resolve_repo_path(
            bert_args.trainer.timevocab_pth))
        self.sizevocab = SizeVocab.load_vocab(resolve_repo_path(
            bert_args.trainer.sizevocab_pth))
        self.dataset = RLFlowDataset(rl_args, self.timevocab, self.sizevocab, mode)
        if not self.dataset:
            raise ValueError("the {} RL dataset is empty".format(mode))

        bert = BERT(len(self.timevocab), len(self.sizevocab),
                    bert_args.model.hidden, bert_args.model.d_ff,
                    bert_args.model.n_layers, bert_args.model.attn_heads,
                    bert_args.model.dropout)
        self.model = BERTLM(bert, len(self.timevocab),
                            len(self.sizevocab)).to(self.device)
        checkpoint = torch.load(resolve_repo_path(rl_args.trainer.bert_pth),
                                map_location=self.device)
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.eval()
        self.data_index = 0
        self.state = None
        self.step_num = 0

    def _observation(self):
        return clone_state(self.state, self.device)

    def reset(self):
        self.step_num = 0
        initial, physical, metadata = self.dataset[self.data_index]
        self.data_index = (self.data_index + 1) % len(self.dataset)
        self.state = {name: value.to(self.device) for name, value in initial.items()}
        self.real_feat = deepcopy(physical)
        self.src = list(metadata["src"])
        self.dst = list(metadata["dst"])
        self.srcport = list(metadata["src_port"])
        self.dstport = list(metadata["dst_port"])
        attacker = self.rl_args.env.bad_ip
        attacker = self.src[0] if attacker is None else str(attacker)
        self.attacker_ip = attacker
        self.src_index = [position + 1 for position, source in enumerate(self.src)
                          if source == attacker]
        if not self.src_index:
            raise ValueError("attacker IP {!r} is absent from flow".format(attacker))
        self._refresh_src_mask()
        return self._observation()

    def _refresh_src_mask(self):
        source_mask = torch.zeros(self.state_dim, dtype=torch.long,
                                  device=self.device)
        valid = [index for index in self.src_index if index < self.state_dim]
        if valid:
            source_mask[valid] = 1
        self.state["src_index"] = source_mask

    @staticmethod
    def _insert_tensor(sequence, value, position):
        value = torch.as_tensor([value], dtype=sequence.dtype,
                                device=sequence.device)
        return torch.cat((sequence[:position], value,
                          sequence[position:]))[:sequence.shape[0]]

    def _predict_tokens(self, masked_ipd, masked_size, position):
        with torch.no_grad():
            ipd_logits, size_logits = self.model(masked_ipd.unsqueeze(0),
                                                  masked_size.unsqueeze(0))
        time_start, time_stop = (self.timevocab.valid_index_start,
                                 self.timevocab.valid_index_stop)
        # Ethernet/IP packets in this dataset are at least 20 bytes.
        size_start = max(self.sizevocab.valid_index_start,
                         self.sizevocab.stoi(20))
        size_stop = self.sizevocab.valid_index_stop
        time_token = int(ipd_logits[0, position, time_start:time_stop]
                         .argmax().item() + time_start)
        size_token = int(size_logits[0, position, size_start:size_stop]
                         .argmax().item() + size_start)
        return time_token, size_token

    def _insertion_metadata(self):
        packet_index = self.src_index[0] - 1
        return (self.attacker_ip, self.dst[packet_index],
                self.srcport[packet_index], self.dstport[packet_index])

    def apply_action(self, action):
        """Apply exactly one legal edit without querying the target model."""
        action = int(action)
        legal = generate_mask(self.state["real_length"], self.state_dim,
                              self.state["src_index"])
        if action < 0 or action >= legal.numel() or not bool(legal[action]):
            raise ValueError("illegal packet-edit action {}".format(action))
        position = action // 2
        ipd = self.state["flow_ipd"].clone()
        size = self.state["flow_size"].clone()

        if action % 2 == 0:
            insertion_metadata = self._insertion_metadata()
            ipd = self._insert_tensor(ipd, self.timevocab.mask_index, position)
            size = self._insert_tensor(size, self.sizevocab.mask_index, position)
            time_token, size_token = self._predict_tokens(ipd, size, position)
            ipd[position], size[position] = time_token, size_token
            self.state["real_length"] += 1
            self.src_index = [index + 1 if index >= position else index
                              for index in self.src_index]
            bisect.insort(self.src_index, position)
            packet_index = position - 1
            source, destination, source_port, destination_port = insertion_metadata
            self.src.insert(packet_index, source)
            self.dst.insert(packet_index, destination)
            self.srcport.insert(packet_index, source_port)
            self.dstport.insert(packet_index, destination_port)
            self.real_feat["ipd"].insert(
                packet_index, self.timevocab.itos(time_token))
            self.real_feat["size"].insert(
                packet_index, self.sizevocab.itos(size_token))
        else:
            masked_ipd, masked_size = ipd.clone(), size.clone()
            masked_ipd[position] = self.timevocab.mask_index
            masked_size[position] = self.sizevocab.mask_index
            time_token, _size_token = self._predict_tokens(
                masked_ipd, masked_size, position)
            ipd[position] = time_token
            self.real_feat["ipd"][position - 1] = \
                self.timevocab.itos(time_token)

        self.state["flow_ipd"] = ipd
        self.state["flow_size"] = size
        self._refresh_src_mask()
        self.step_num += 1
        return self._observation()

    def get_res(self):
        raise NotImplementedError

    def step(self, action):
        observation = self.apply_action(action)
        detected = self.get_res()
        evaded = detected == 0
        done = evaded or self.step_num >= int(self.rl_args.trainer.max_stop_step)
        reward = 1.0 if evaded else -0.03
        return observation, reward, done, evaded


class NetBeaconEnv(BaseEnv):
    def __init__(self, rl_args, bert_args, mode="train"):
        super().__init__(rl_args, bert_args, mode)
        self.evaluator = NetBeaconEvaluator(resolve_repo_path(
            rl_args.trainer.target_model_pth))

    def get_res(self):
        return self.evaluator.evaluate(self.real_feat["ipd"],
                                       self.real_feat["size"])


ENV_REGISTRY = {"NetBeacon": NetBeaconEnv}


def make_env(rl_args, bert_args, mode="train"):
    name = rl_args.trainer.env_name
    if name not in ENV_REGISTRY:
        raise ValueError("unknown env_name {!r}; available: {}".format(
            name, ", ".join(sorted(ENV_REGISTRY))))
    return ENV_REGISTRY[name](rl_args, bert_args, mode)
