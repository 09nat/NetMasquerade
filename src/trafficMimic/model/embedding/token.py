import torch
import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size):
        super(TokenEmbedding, self).__init__(vocab_size, embed_size, padding_idx=0)