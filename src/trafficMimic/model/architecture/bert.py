import torch
import torch.nn as nn
import numpy as np
from .Encoder import EncoderBlock
from ..embedding import BERTEmbedding


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, timevocab_size, sizevocab_size, hidden=768, d_ff=768 * 4, n_layers=12, attn_heads=8, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4 * hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = d_ff

        # embedding for BERT, sum of positional, segment, token embeddings
        self.ipd_embedding = BERTEmbedding(vocab_size=timevocab_size, embed_size=hidden)
        self.size_embedding = BERTEmbedding(vocab_size=sizevocab_size, embed_size=hidden)

        # multi-layers transformer blocks, deep network
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])

    def forward(self, flow_ipd, flow_size):
        # x: [bs, seq_len]
        # segment_info: [bs, seq_len]
        # print('flow_ipd.max(),', end='\t')
        # print(flow_ipd.max())
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (flow_size > 0).unsqueeze(1).repeat(1, flow_size.size(1), 1).unsqueeze(1)
        
        # embedding the indexed sequence to sequence of vectors
        flow_ipd = self.ipd_embedding(flow_ipd)
        # print('flow_ipd: ', flow_ipd)
        flow_size = self.size_embedding(flow_size)
        # print('flow_size: ', flow_size)

        # running over multiple transformer blocks
        for i, encoder in enumerate(self.encoder_blocks):
            flow_ipd, flow_size = encoder.forward(flow_ipd, flow_size, mask)    # literally ipd_attn, size_attn
            # print('flow_size_attention_', i, ': ', flow_size)
        return flow_ipd, flow_size


