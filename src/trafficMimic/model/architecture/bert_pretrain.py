import torch.nn as nn

from .bert import BERT


class BERTLM(nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert, timevocab_size=2000, sizevocab_size=2000):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
        self.mask_lm = MaskedLanguageModel(self.bert.hidden, timevocab_size, sizevocab_size)

    def forward(self, flow_ipd, flow_size):
        flow_ipd, flow_size = self.bert(flow_ipd, flow_size)

        return self.mask_lm(flow_ipd, flow_size)



class NextSentencePrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))


class MaskedLanguageModel(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, timevocab_size, sizevocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear_1 = nn.Linear(hidden, timevocab_size)
        self.linear_2 = nn.Linear(hidden, sizevocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, flow_ipd, flow_size):
        return self.softmax(self.linear_1(flow_ipd)), self.softmax(self.linear_2(flow_size))