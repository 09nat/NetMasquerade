"""Fixed vocabularies used by the released traffic-BERT checkpoint."""

import pickle

import numpy as np


class _VocabUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in {"dataset.vocab", "vocab"} and name in globals():
            return globals()[name]
        return super().find_class(module, name)


class Vocab:
    def __init__(self, data, max_size):
        self.special_list = ["[pad]", "[unk]", "[cls]", "[sep]", "[mask]"]
        self.special_size = len(self.special_list)
        self.reverse_list = {token: index for index, token in enumerate(self.special_list)}
        self.pad_index = 0
        self.unk_index = 1
        # These historical IDs are intentionally retained for checkpoint compatibility.
        self.eos_index = 2
        self.sos_index = 3
        self.mask_index = 4
        self.data = data
        self.max_size = max_size + self.special_size

    def __len__(self):
        # The original checkpoint contains one unused final embedding row.
        return self.max_size + 1

    @property
    def valid_index_start(self):
        return self.special_size

    @property
    def valid_index_stop(self):
        return self.max_size

    @classmethod
    def load_vocab(cls, vocab_path):
        with open(vocab_path, "rb") as handle:
            vocab = _VocabUnpickler(handle).load()
        if not isinstance(vocab, Vocab):
            raise TypeError("{} does not contain a Vocab".format(vocab_path))
        return vocab

    def save_vocab(self, vocab_path):
        original_data = self.data
        self.data = None
        try:
            with open(vocab_path, "wb") as handle:
                pickle.dump(self, handle, protocol=4)
        finally:
            self.data = original_data

    def from_seq(self, sequence, with_pad=False):
        sequence = list(sequence)
        if not with_pad:
            while sequence and sequence[-1] == self.pad_index:
                sequence.pop()
        return [self.itos(index) for index in sequence]


class TimeVocab(Vocab):
    def __init__(self, data, max_size=50, upper_bound=300, lower_bound=None):
        super().__init__(data, max_size)
        self.upper_bound = upper_bound if upper_bound is not None else max(
            max(item[1:]) for item in data)
        self.lower_bound = lower_bound if lower_bound is not None else min(
            min(item[1:]) for item in data)
        self.bins = self.binning()

    def binning(self):
        minimum = np.log10(self.lower_bound + 1e-9)
        maximum = np.log10(self.upper_bound + 1e-9)
        return np.linspace(minimum, maximum,
                           self.max_size - self.special_size,
                           endpoint=False)

    def itos(self, index):
        index = int(index)
        if index < self.special_size:
            return self.special_list[index]
        if not self.valid_index_start <= index < self.valid_index_stop:
            return self.special_list[self.unk_index]
        bucket = index - self.special_size
        left = self.bins[bucket]
        right = (self.bins[bucket + 1] if bucket + 1 < len(self.bins)
                 else np.log10(self.upper_bound + 1e-9))
        return float(np.power(10.0, (left + right) / 2.0))

    def stoi(self, value):
        if isinstance(value, str) and value in self.reverse_list:
            return self.reverse_list[value]
        value = float(value)
        if value < self.lower_bound or value >= self.upper_bound:
            return self.unk_index
        bucket = int(np.digitize(np.log10(value + 1e-9), self.bins)) - 1
        bucket = min(max(bucket, 0), len(self.bins) - 1)
        return bucket + self.special_size

    def to_seq(self, sentence, seq_len, with_eos=False, with_sos=False,
               with_ori_len=False):
        sequence = [self.stoi(value) for value in sentence]
        return _finish_sequence(self, sequence, seq_len, with_eos, with_sos,
                                with_ori_len)


class SizeVocab(Vocab):
    def __init__(self, data, ratio=1):
        self.ratio = ratio
        super().__init__(data, 1600 // ratio)

    def hash(self, value):
        return int(value) // self.ratio

    def rehash(self, index):
        return int(index * self.ratio)

    def itos(self, index):
        index = int(index)
        if index < self.special_size:
            return self.special_list[index]
        if not self.valid_index_start <= index < self.valid_index_stop:
            return self.special_list[self.unk_index]
        return self.rehash(index - self.special_size)

    def stoi(self, value):
        if isinstance(value, str) and value in self.reverse_list:
            return self.reverse_list[value]
        index = self.hash(value) + self.special_size
        return (index if self.valid_index_start <= index < self.valid_index_stop
                else self.unk_index)

    def to_seq(self, sentence, seq_len, with_eos=False, with_sos=False,
               with_ori_len=False):
        sequence = [self.stoi(value) for value in sentence]
        return _finish_sequence(self, sequence, seq_len, with_eos, with_sos,
                                with_ori_len)


def _finish_sequence(vocab, sequence, seq_len, with_eos, with_sos, with_ori_len):
    if with_eos:
        sequence.append(vocab.eos_index)
    if with_sos:
        sequence.insert(0, vocab.sos_index)
    original_length = len(sequence)
    if seq_len is not None:
        sequence = sequence[:seq_len]
        sequence.extend([vocab.pad_index] * (seq_len - len(sequence)))
    return (sequence, original_length) if with_ori_len else sequence
