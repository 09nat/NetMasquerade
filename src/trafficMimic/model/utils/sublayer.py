import torch.nn as nn
from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(size)
        self.norm_2 = LayerNorm(size)

    def forward(self, x, sublayer, y=None):
        if y is None:
            return x + self.dropout(sublayer(self.norm(x)))
        else:
            return x + self.dropout(sublayer(self.norm(y), self.norm_2(x)))
