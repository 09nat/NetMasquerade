"""Turn flow pickles into immutable RL observations and physical features."""

import torch
from torch.utils.data import Dataset

from trafficMimic.utils import (feature_extract_full, read_flow_pkl,
                                resolve_repo_path, valid_flows)


class RLFlowDataset(Dataset):
    """A flow dataset shared by training and held-out evaluation.

    Token positions 0 and ``real_length - 1`` are the historical start/end
    tokens.  Physical IPDs and sizes contain only packet positions, so their
    maximum length is ``state_dim - 2``.
    """

    def __init__(self, args, timevocab, sizevocab, mode="train"):
        self.state_dim = int(args.model.state_dim)
        if self.state_dim < 4:
            raise ValueError("state_dim must be at least 4")
        field = "train_data_pth" if mode == "train" else "test_data_pth"
        self.flows = valid_flows(read_flow_pkl(
            resolve_repo_path(getattr(args.trainer, field))))
        extracted = feature_extract_full(self.flows)
        (self.ipds, self.sizes, self.sources, self.destinations,
         self.source_ports, self.destination_ports) = extracted
        self.timevocab = timevocab
        self.sizevocab = sizevocab

    def __len__(self):
        return len(self.flows)

    def __getitem__(self, index):
        payload_limit = self.state_dim - 2
        ipd = list(self.ipds[index][:payload_limit])
        size = list(self.sizes[index][:payload_limit])
        time_tokens = ([self.timevocab.sos_index] +
                       self.timevocab.to_seq(ipd, None) +
                       [self.timevocab.eos_index])
        size_tokens = ([self.sizevocab.sos_index] +
                       self.sizevocab.to_seq(size, None) +
                       [self.sizevocab.eos_index])
        real_length = len(time_tokens)
        time_tokens.extend([self.timevocab.pad_index] *
                           (self.state_dim - real_length))
        size_tokens.extend([self.sizevocab.pad_index] *
                           (self.state_dim - real_length))
        state = {
            "flow_ipd": torch.tensor(time_tokens, dtype=torch.long),
            "flow_size": torch.tensor(size_tokens, dtype=torch.long),
            "real_length": torch.tensor(real_length, dtype=torch.long),
        }
        physical = {"ipd": ipd, "size": size}
        metadata = {
            "src": list(self.sources[index][:payload_limit]),
            "dst": list(self.destinations[index][:payload_limit]),
            "src_port": list(self.source_ports[index][:payload_limit]),
            "dst_port": list(self.destination_ports[index][:payload_limit]),
        }
        return state, physical, metadata
