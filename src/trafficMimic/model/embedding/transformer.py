import torch.nn as nn
from .position import PositionalEmbedding
from .token import TokenEmbedding


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size=768, dropout=0.1):
        super(TransformerEmbedding, self).__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size, )
        self.position = PositionalEmbedding(self.token.embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)
