import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import torch
from Pretrain.utils import *
from torch.utils.data import Dataset, DataLoader
from Pretrain.dataset.vocab import *
from copy import deepcopy
# max_size: 7340, vocab: 7340

class RLFlowDataset(Dataset):
    def __init__(self, args, timevocab, sizevocab, mode) -> None:
        super(Dataset, self).__init__()
        self.seq_len = args.model.state_dim
        if mode == 'train':
            self.data = list(feature_extract(read_flow_pkl(args.trainer.train_data_pth)))
        else:
            self.data = list(feature_extract(read_flow_pkl(args.trainer.test_data_pth)))
        self.timevocab = timevocab
        self.sizevocab = sizevocab

        self.ipd_list, self.size_list = list(self.data[0]), list(self.data[1])

    def __len__(self, ):
        return len(self.ipd_list)

    def __getitem__(self, item):
        flow_ipd = self.timevocab.to_seq(self.ipd_list[item], None)
        flow_size = self.sizevocab.to_seq(self.size_list[item], None)

        flow_ipd = [self.timevocab.sos_index] + flow_ipd + [self.timevocab.eos_index]
        flow_size = [self.sizevocab.sos_index] + flow_size + [self.sizevocab.eos_index]

        real_length = len(flow_size)

        flow_ipd = flow_ipd[:self.seq_len]
        flow_size = flow_size[:self.seq_len]

        padding = [self.timevocab.pad_index for _ in range(self.seq_len - len(flow_ipd))]
        flow_ipd.extend(padding), flow_size.extend(padding)

        # print(len(flow_ipd), len(flow_ipd_label), len(flow_size), len(flow_size_label))
        output = {"flow_ipd": flow_ipd,
                  "flow_size": flow_size,
                  "real_length": real_length}

        return {key: torch.tensor(value) for key, value in output.items()}
    

class RLFullFlowDataset(Dataset):
    def __init__(self, args, timevocab, sizevocab, mode) -> None:
        super(Dataset, self).__init__()
        self.seq_len = args.model.state_dim
        if mode == 'train':
            self.flow_list = read_flow_pkl(args.trainer.train_data_pth)
            self.data = list(feature_extract_full(self.flow_list))
        else:
            self.flow_list = read_flow_pkl(args.trainer.test_data_pth)
            self.data = list(feature_extract_full(self.flow_list))
        self.timevocab = timevocab
        self.sizevocab = sizevocab
        self.ipd_list, self.size_list, self.src, self.dst, self.src_port, self.dst_port = list(self.data[0]), list(self.data[1]), list(self.data[2]), list(self.data[3]), list(self.data[4]), list(self.data[5])

    def __len__(self, ):
        return len(self.ipd_list)

    def __getitem__(self, item):
        flow_ipd = self.timevocab.to_seq(self.ipd_list[item], None)
        flow_size = self.sizevocab.to_seq(self.size_list[item], None)

        flow_ipd = [self.timevocab.sos_index] + flow_ipd + [self.timevocab.eos_index]
        flow_size = [self.sizevocab.sos_index] + flow_size + [self.sizevocab.eos_index]

        real_length = len(flow_size)

        flow_ipd = flow_ipd[:self.seq_len]
        flow_size = flow_size[:self.seq_len]

        padding = [self.timevocab.pad_index for _ in range(self.seq_len - len(flow_ipd))]
        flow_ipd.extend(padding), flow_size.extend(padding)

        output = {"flow_ipd": flow_ipd,
                  "flow_size": flow_size,
                  "real_length": real_length}

        ori_output = {
            "ipd": self.ipd_list[item][:self.seq_len - 1],
            "size": self.size_list[item][:self.seq_len - 1],
        }
        return {key: torch.tensor(value) for key, value in output.items()}, ori_output, self.src[item], self.dst[item], self.src_port[item], self.dst_port[item], self.flow_list[item]

    def collate_fn(self, batch):
        return batch
    
    
class RLFullFlowNoDiffDataset(Dataset):
    def __init__(self, args, timevocab, sizevocab, mode) -> None:
        super(Dataset, self).__init__()
        self.seq_len = args.model.state_dim
        if mode == 'train':
            self.flow_list = read_flow_pkl(args.trainer.train_data_pth)
            self.data = list(feature_extract_full_wo_diff(self.flow_list))
        else:
            self.flow_list = read_flow_pkl(args.trainer.test_data_pth)
            self.data = list(feature_extract_full_wo_diff(self.flow_list))
        self.timevocab = timevocab
        self.sizevocab = sizevocab
        self.ipd_list, self.size_list, self.src, self.dst, self.src_port, self.dst_port = list(self.data[0]), list(self.data[1]), list(self.data[2]), list(self.data[3]), list(self.data[4]), list(self.data[5])

    def __len__(self, ):
        return len(self.ipd_list)

    def __getitem__(self, item):
        # print(self.ipd_list[item])
        stp = deepcopy(self.ipd_list[item])
        for i in range(len(self.ipd_list[item]) - 1, 0, -1):
            stp[i] = stp[i] - stp[i - 1]
        stp[0] = 0
        
        flow_ipd = self.timevocab.to_seq(stp, None)
        flow_size = self.sizevocab.to_seq(self.size_list[item], None)

        flow_ipd = [self.timevocab.sos_index] + flow_ipd + [self.timevocab.eos_index]
        flow_size = [self.sizevocab.sos_index] + flow_size + [self.sizevocab.eos_index]

        real_length = len(flow_size)

        flow_ipd = flow_ipd[:self.seq_len]
        flow_size = flow_size[:self.seq_len]

        padding = [self.timevocab.pad_index for _ in range(self.seq_len - len(flow_ipd))]
        flow_ipd.extend(padding), flow_size.extend(padding)

        output = {"flow_ipd": flow_ipd,
                  "flow_size": flow_size,
                  "real_length": real_length}

        ori_output = {
            # "ipd": self.ipd_list[item][:self.seq_len - 1],
            "ipd": stp[:self.seq_len - 1],
            "size": self.size_list[item][:self.seq_len - 1],
        }
        return {key: torch.tensor(value) for key, value in output.items()}, ori_output, self.src[item], self.dst[item], self.src_port[item], self.dst_port[item], self.flow_list[item]

    def collate_fn(self, batch):
        return batch

if __name__ == '__main__':
    args = recursive_namespace(read_yaml('/home/lzx/NetMasquerade/Pretrain/config/bert.yaml'))
    rl_args = recursive_namespace(read_yaml('/home/lzx/NetMasquerade/Finetune/config/sac.yaml'))
    data = list(feature_extract(read_flow_pkl(rl_args.trainer.train_data_pth)))
    ipd, size = list(data[0]), list(data[1])
    timevocab = TimeVocab(ipd)
    sizevocab = SizeVocab(size)

    data_iter = 0
    train_dataset = RLFullFlowDataset(rl_args, timevocab, sizevocab, mode='test')
    for _ in range(400):
        initial_state, real_feat, src, dst, srcport, dstport = train_dataset[data_iter]
        data_iter = data_iter + 1 if data_iter < len(train_dataset) - 1 else 0
        print(data_iter, src)

