import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import tqdm
import pickle
import numpy as np
from utils import recursive_namespace, feature_extract, read_yaml, read_flow_pkl


class Vocab(object):
    def __init__(self, data, max_size):
        '''
        no use of '[cls]', '[sep]' in this project
        '''
        self.special_list = ['[pad]', '[unk]', '[cls]', '[sep]', '[mask]']
        self.special_size = len(self.special_list)
        self.reverse_list = {token: i for i, token in enumerate(self.special_list)}
        self.pad_index = 0
        self.unk_index = 1
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        self.data = data
        self.max_size = max_size + self.special_size

    def __len__(self):
        return self.max_size + 1

    def to_seq(self, sentence, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    def from_seq(self, seq, with_pad=False):
        pass

    def itos(self, index):
        pass

    def stoi(self, word):
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, 'rb') as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        original_data = self.data
        self.data = None
        with open(vocab_path, 'wb') as f:
            pickle.dump(self, f)
        self.data = original_data


class TimeVocab(Vocab):
    def __init__(self, data, max_size=50, upper_bound=300, lower_bound=None):
        super(TimeVocab, self).__init__(data, max_size)
        if upper_bound is not None:
            self.upper_bound = upper_bound
        else:
            self.upper_bound = max(max(i[1:]) for i in self.data)
            
        if lower_bound is not None:
            self.lower_bound = lower_bound
        else:
            self.lower_bound = min(min(i[1:]) for i in self.data)
        self.bins = self.binning()
        

    def binning(self, ):
        min_val = np.log10(self.lower_bound + 1e-9)
        max_val = np.log10(self.upper_bound + 1e-9)
        bin_width = (max_val - min_val) / (self.max_size - self.special_size)
        bins = np.arange(min_val, max_val, bin_width)

        return bins
    

    def itos(self, index):
        if index < self.special_size:
            return self.special_list[index]
        if index >= self.max_size:
            return self.unk_index
        if index == self.max_size - 1:
            return self.bins[-1]
        index -= self.special_size
        return np.power(10, (self.bins[index + 1] + self.bins[index]) / 2.0)


    def stoi(self, word):
        if word in self.reverse_list:
            return self.reverse_list[word]
        if word + 1e-9 >= self.upper_bound:
            return self.unk_index
        if word + 1e-9 < self.bins[0]:
            return self.unk_index
        return np.digitize(np.log10(word + 1e-9), self.bins) + self.special_size - 1

    def to_seq(self, sentence, seq_len, with_eos=False, with_sos=False, with_ori_len=False) -> list:
        seq = [self.stoi(word) for word in sentence]

        if with_eos:
            seq += [self.eos_index]
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_ori_len else seq

    def from_seq(self, seq, with_pad=False):
        sentence = [self.itos(index) for index in seq]
        cnt = len(sentence)
        if not with_pad:
            while cnt > 0 and sentence[cnt - 1] == 0:
                cnt -= 1
        return sentence[: cnt]


class SizeVocab(Vocab):
    def __init__(self, data, ratio=1):
        self.ratio = ratio
        max_size = 1600 // ratio
        super(SizeVocab, self).__init__(data, max_size)

    def hash(self, word):
        return int(word) // self.ratio
    
    def rehash(self, index):
        return int(index * self.ratio)

    def itos(self, index):
        if index < self.special_size:
            return self.special_list[index]
        if index > self.max_size:
            return self.special_list[self.unk_index]
        return self.rehash(index - self.special_size)

    def stoi(self, word):
        if word in self.reverse_list:
            return self.reverse_list[word]
        if self.hash(word) + self.special_size >= self.max_size:
            return self.unk_index
        return self.hash(word) + self.special_size

    def to_seq(self, sentence, seq_len, with_eos=False, with_sos=False, with_ori_len=False) -> list:
        seq = [self.stoi(word) for word in sentence]

        if with_eos:
            seq += [self.eos_index]
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_ori_len else seq

    def from_seq(self, seq, with_pad=False):
        sentence = [self.itos(index) for index in seq]
        cnt = len(sentence)
        if not with_pad:
            while cnt > 0 and sentence[cnt - 1] == 0:
                cnt -= 1
        return sentence[: cnt]


if __name__ == '__main__':
    pass