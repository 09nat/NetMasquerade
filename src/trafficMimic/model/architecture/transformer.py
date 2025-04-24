import torch
import torch.nn as nn
import numpy as np
from .Encoder import EncoderBlock
from .Decoder import DecoderBlock
from ..embedding import TransformerEmbedding


class Transformer(nn.Module):
    def __init__(self, vocab_size, hidden=768, d_ff=768 * 4, n_layers=12, attn_heads=12, dropout=0.1):
        """
        :param vocab_size: vocab_size of total words
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super(Transformer, self).__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4 * hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = d_ff

        # embedding for BERT, sum of positional, segment, token embeddings
        self.enc_embedding = TransformerEmbedding(vocab_size=vocab_size, embed_size=hidden)
        self.dec_embedding = TransformerEmbedding(vocab_size=vocab_size, embed_size=hidden)
        # multi-layers transformer blocks, deep network
        self.encoder_blocks = nn.ModuleList(
            [EncoderBlock(hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])

        self.decoder_blocks = nn.ModuleList(
            [DecoderBlock(hidden, attn_heads, self.feed_forward_hidden, dropout) for _ in range(n_layers)])

    def forward(self, enc, dec):
        # x: [bs, seq_len]
        # segment_info: [bs, seq_len]

        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        enc_mask = (enc > 0).unsqueeze(1).repeat(1, enc.size(1), 1).unsqueeze(1)
        dec_pad_mask = (dec > 0).unsqueeze(1).repeat(1, dec.size(1), 1).unsqueeze(1)
        dec_subsequence_mask = torch.from_numpy(np.triu(np.ones([dec.size(0), dec.size(1), dec.size(1)]), k=1)).byte()
        dec_mask = torch.gt((dec_pad_mask + dec_subsequence_mask),0)  # [batch_size, tgt_len, tgt_len]
        dec_enc_mask = (enc > 0).unsqueeze(1).repeat(1, dec.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        enc_output = self.enc_embedding(enc)
        dec_output = self.dec_embedding(dec)

        for encoder in self.encoder_blocks:
            enc_output = encoder.forward(enc_output, enc_mask)

        for decoder in self.decoder_blocks:
            dec_output = decoder(enc_output, dec_output, dec_mask, dec_enc_mask)

        return dec_output
