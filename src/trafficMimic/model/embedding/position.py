import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        # position: [max_len, 1], 1 for broadcast
        position = torch.arange(0, max_len).float().unsqueeze(1)
        # div_term: [max_len / 2, ]
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # pe(pos, 2i) = sin(pos / 10000 ^ (2i / d_model))
        # pe(pos, 2i + 1) = cos(pos / 10000 ^ (2i / d_model))
        # pe: [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
