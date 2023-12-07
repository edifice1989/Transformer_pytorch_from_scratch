import torch
import torch.nn as nn
from ..layers.EncoderLayer import EncoderLayer
from ..func.clones import clones
from ..func.LayerNorm import LayerNorm

class Encoder(nn.Module):

    def __init__(self,layer,N):

        super(Encoder,self).__init__()
       
        # n_copy = 6 
        self.norm = LayerNorm(layer.size)
        self.layers = clones(layer,N)
                           

    def forward(self, x,src_mask):

        for layer in self.layers:

            x = layer(x,src_mask)

        return self.norm(x)