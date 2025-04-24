import torch
import torch.nn as nn


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=3):
        super(SegmentEmbedding, self).__init__(3, embed_size, padding_idx=0)

    # forward:
    # e.g. sentences [bs, seq_len]
    #      segment_label [bs, seq_len] (1 or 2) or (0 or 1)
