import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import torch
from torch.utils.data import Dataset, DataLoader
from utils import feature_extract, read_flow_pkl
from dataset.vocab import *
# max_size: 7340, vocab: 7340

class FlowDataset(Dataset):
    def __init__(self, args, timevocab, sizevocab, train=False) -> None:
        super(Dataset, self).__init__()
        self.seq_len = args.model.seq_len
        if train:
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

        flow_ipd_random, flow_ipd_label = self.random_word(flow_ipd, self.timevocab)
        flow_size_random, flow_size_label = self.random_word(flow_size, self.sizevocab)

        flow_ipd = [self.timevocab.sos_index] + flow_ipd_random + [self.timevocab.eos_index]
        flow_size = [self.sizevocab.sos_index] + flow_size_random + [self.sizevocab.eos_index]

        real_length = len(flow_size)

        flow_ipd_label = [self.timevocab.pad_index] + flow_ipd_label + [self.timevocab.pad_index]
        flow_size_label = [self.sizevocab.pad_index] + flow_size_label + [self.sizevocab.pad_index]

        flow_ipd = flow_ipd[:self.seq_len]
        flow_size = flow_size[:self.seq_len]
        flow_ipd_label = flow_ipd_label[:self.seq_len]
        flow_size_label = flow_size_label[:self.seq_len]

        padding = [self.timevocab.pad_index for _ in range(self.seq_len - len(flow_ipd))]
        flow_ipd.extend(padding), flow_size.extend(padding), flow_ipd_label.extend(padding), flow_size_label.extend(padding)

        # print(len(flow_ipd), len(flow_ipd_label), len(flow_size), len(flow_size_label))
        output = {"flow_ipd": flow_ipd,
                  "flow_size": flow_size,
                  "flow_ipd_label": flow_ipd_label,
                  "flow_size_label": flow_size_label,
                  "real_length": real_length}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, tokens, vocab):
        output_label = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = vocab.mask_index

                # 10% randomly change token to random token
                elif prob < 0.9:
                    # tokens[i] = random.randrange(len(vocab))
                    tokens[i] = random.randint(5, vocab.__len__() - 1)

                # 10% ratndomly change token to current token
                else:
                    tokens[i] = vocab.stoi(token)

                output_label.append(token)
                # print('token:\t', token)

            else:
                tokens[i] = token
                output_label.append(0)
        return tokens, output_label



if __name__ == '__main__':
    args = recursive_namespace(read_yaml('/home/lzx/NetMasquerade/Pretrain/config/bert.yaml'))
    data = list(feature_extract(read_flow_pkl(args.trainer.train_data_pth)))
    ipd, size = list(data[0]), list(data[1])
    timevocab = TimeVocab(ipd)
    sizevocab = SizeVocab(size)

    train_dataset = FlowDataset(args, timevocab, sizevocab, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.trainer.batch_size,shuffle=True, drop_last=True)

    cnt = 0
    for batch in train_dataloader:
        print('batch[flow_ipd_label]: ', batch['flow_ipd_label'])

