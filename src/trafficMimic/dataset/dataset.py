"""Masked-language-model dataset for optional traffic-BERT retraining."""

import random

import torch
from torch.utils.data import Dataset

from trafficMimic.utils import feature_extract, read_flow_pkl, valid_flows


class FlowDataset(Dataset):
    def __init__(self, args, timevocab, sizevocab, train=False):
        self.seq_len = args.model.seq_len
        path = (args.trainer.train_data_pth if train
                else args.trainer.test_data_pth)
        flows = valid_flows(read_flow_pkl(path))
        self.ipd_list, self.size_list = feature_extract(flows)
        self.timevocab = timevocab
        self.sizevocab = sizevocab

    def __len__(self):
        return len(self.ipd_list)

    def __getitem__(self, item):
        max_payload = self.seq_len - 2
        ipd = self.timevocab.to_seq(self.ipd_list[item][:max_payload], None)
        size = self.sizevocab.to_seq(self.size_list[item][:max_payload], None)
        ipd, ipd_label = self.random_word(ipd, self.timevocab)
        size, size_label = self.random_word(size, self.sizevocab)

        ipd = [self.timevocab.sos_index] + ipd + [self.timevocab.eos_index]
        size = [self.sizevocab.sos_index] + size + [self.sizevocab.eos_index]
        ipd_label = [0] + ipd_label + [0]
        size_label = [0] + size_label + [0]
        real_length = len(size)

        padding = self.seq_len - real_length
        ipd.extend([self.timevocab.pad_index] * padding)
        size.extend([self.sizevocab.pad_index] * padding)
        ipd_label.extend([0] * padding)
        size_label.extend([0] * padding)
        output = {
            "flow_ipd": ipd,
            "flow_size": size,
            "flow_ipd_label": ipd_label,
            "flow_size_label": size_label,
            "real_length": real_length,
        }
        return {key: torch.tensor(value, dtype=torch.long)
                for key, value in output.items()}

    @staticmethod
    def random_word(tokens, vocab):
        labels = []
        for index, token in enumerate(tokens):
            probability = random.random()
            if probability >= 0.15:
                labels.append(0)
                continue
            labels.append(token)
            probability /= 0.15
            if probability < 0.8:
                tokens[index] = vocab.mask_index
            elif probability < 0.9:
                tokens[index] = random.randrange(vocab.valid_index_start,
                                                  vocab.valid_index_stop)
            # final 10% intentionally keeps the already-tokenized ID unchanged
        return tokens, labels
