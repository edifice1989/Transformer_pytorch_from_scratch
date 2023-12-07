import torch
import torch.nn as nn
from models.func import self_attention
from models.func import SublayerConnection

from models.layers import FeedForwardLayer

from models.layers import MultiHeadAttention


from models.func import clones

class EncoderLayer(nn.Module):

    def __init__(self,size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()

        self.multi_head_attn = MultiHeadAttention
        self.feed_forward = FeedForwardLayer
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x,mask):
    
        
        # we need to pass a func layer to sublayer as the 2nd parameter NOT the computational result
        # so we use lambda func here to map one input x to x,x,x 
      
        x = self.sublayer[0](x, lambda x: self.multi_head_attn(x, x, x, mask)) 


        return self.sublayer[1](x, self.feed_forward)