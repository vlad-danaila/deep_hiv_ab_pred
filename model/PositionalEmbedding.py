import torch
from torch import nn
from torch import Tensor
import math
import matplotlib.pyplot as plt
from deep_hiv_ab_pred.util.tools import to_numpy, to_torch

# implementation adaptation from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
# for virus good values are 6 and 1019
# for ab good values are 4 and 227
def get_positional_embeding(pos_embed_size, seq_len):
    pos = torch.arange(seq_len).unsqueeze(1)
    div = torch.exp(torch.arange(0, pos_embed_size, 2) * (-math.log(10000.0) / pos_embed_size))
    pe = torch.zeros(seq_len, pos_embed_size)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

if __name__ == '__main__':
    pe_virus = to_numpy(get_positional_embeding(6, 1019))
    pe_ab = to_numpy(get_positional_embeding(4, 227))