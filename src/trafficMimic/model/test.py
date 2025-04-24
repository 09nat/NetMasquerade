import torch
import numpy as np
from architecture import BERT, Transformer

sequence = torch.from_numpy(np.random.randint(1, 19, (12, 8)))
sequence_label = torch.tensor(np.array([0 for _ in range(4)] + [1 for _ in range(4)]))
bert = BERT(20)
output = bert(sequence, sequence_label)
print(output)

transformer = Transformer(20)
output = transformer(sequence, sequence)
print(output)
