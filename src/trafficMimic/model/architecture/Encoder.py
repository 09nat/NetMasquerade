import torch.nn as nn

from ..attention import MultiHeadedAttention
from ..utils import SublayerConnection, PositionwiseFeedForward


class EncoderBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """
        super(EncoderBlock, self).__init__()
        self.ipd_self_attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.size_self_attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.ipd_size_attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.size_ipd_attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)

        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        
        self.self_attn_linear_1 = SublayerConnection(size=hidden, dropout=dropout)
        self.self_attn_linear_2 = SublayerConnection(size=hidden, dropout=dropout)

        self.input_sublayer_1 = SublayerConnection(size=hidden, dropout=dropout)
        self.input_sublayer_2 = SublayerConnection(size=hidden, dropout=dropout)
        self.input_sublayer_3 = SublayerConnection(size=hidden, dropout=dropout)
        self.input_sublayer_4 = SublayerConnection(size=hidden, dropout=dropout)

        self.output_sublayer_1 = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer_2 = SublayerConnection(size=hidden, dropout=dropout)

        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, flow_ipd, flow_size, mask):
        ipd_self_attn = self.input_sublayer_1(flow_ipd, lambda _x: self.ipd_self_attention.forward(_x, _x, _x, mask=mask))
        ipd_self_attn = self.self_attn_linear_1(ipd_self_attn, self.feed_forward)
        
        
        size_self_attn = self.input_sublayer_3(flow_size, lambda _x: self.size_self_attention.forward(_x, _x, _x, mask=mask))
        size_self_attn = self.self_attn_linear_2(size_self_attn, self.feed_forward)
        
        
        ipd_size_attn = self.input_sublayer_2(ipd_self_attn, lambda _x, _y: self.ipd_size_attention.forward(_y, _x, _x, mask=mask),
                                              flow_size)
        size_ipd_attn = self.input_sublayer_4(size_self_attn, lambda _x, _y: self.size_ipd_attention.forward(_y, _x, _x, mask=mask),
                                              flow_ipd)

        ipd_attn = self.dropout_1(self.output_sublayer_1(ipd_size_attn, self.feed_forward))
        size_attn = self.dropout_2(self.output_sublayer_2(size_ipd_attn, self.feed_forward))
        
    
        return ipd_attn, size_attn
