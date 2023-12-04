import torch
import torch.nn as nn
from models.layers import multi_head_attention
from models.layers import FeedForwardLayer
class Encoder_layer(nn.Module):

    def __init__(self,n_head,d_model,hidden):
        super(Encoder_layer, self).__init__()

        self.norm = nn.LayerNorm(layer.size)

        self.attention_layer= multi_head_attention(d_model, n_head)

        self.feed_forward_layer= FeedForwardLayer(d_model, hidden)

    def forward(self, x):
        # we make a copy for later residue adding
        _x = x
        
        # use multi-head attention we defined in part 1
        atten = self.attention_layer(x)

        # add residue and normalize layer
        _atten = _x + self.norm(atten)

        # feed forward layer which we will define later 
        x = self.feed_forward_layer(x)

        return self.norm(x)+_atten