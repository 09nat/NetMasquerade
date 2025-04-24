import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        '''
        query: [batch_size, n_heads, len_q, d_k]
        key: [batch_size, n_heads, len_k, d_k]
        value: [batch_size, n_heads, len_v(=len_k), d_v]
        mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            # scores = scores.masked_fill(mask == 0, -1e9)
            scores.masked_fill_(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
