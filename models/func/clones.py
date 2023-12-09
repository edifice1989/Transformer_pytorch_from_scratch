import copy
import torch
import torch.nn as nn

def clones(module, N):
   
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])