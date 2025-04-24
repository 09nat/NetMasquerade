import torch.nn as nn
from ..attention import MultiHeadedAttention
from ..utils import SublayerConnection, PositionwiseFeedForward


class DecoderBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """
        super(DecoderBlock, self).__init__()
        self.dec_attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.dec_enc_attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.input_sublayer_2 = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, enc_output, dec, dec_mask, dec_enc_mask):
        # [bs, seq_len, d_model] --> [bs, seq_len, d_model]
        output = self.input_sublayer(dec, lambda _x,: self.dec_attention.forward(_x, _x, _x, mask=dec_mask))
        output = self.input_sublayer_2(output, lambda _x, _y: self.dec_enc_attention.forward(_x, _y, _y, mask=dec_enc_mask), enc_output)

        # [bs, seq_len, d_model] --> [bs, seq_len, d_model]
        x = self.output_sublayer(output, self.feed_forward)
        return self.dropout(x)
