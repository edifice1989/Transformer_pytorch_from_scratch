import torch
import torch.nn as nn
from models.layers import encoder_layer
from models.layers import clones
class Encoder(nn.Module):

    def __init__(self, d_model, hidden, n_head, n_copy):

        super().__init__()
       
        # n_copy = 6 

        self.layers = clones(
                            encoder_layer(d_model,
                                         hidden,
                                         n_head),
                             n_copy)

    def forward(self, x):

        for layer in self.layers:

            x = layer(x)

        return x