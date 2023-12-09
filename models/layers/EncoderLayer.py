import torch
import torch.nn as nn

from models.func.SublayerConnection import SublayerConnection

from models.layers.MultiHeadAttention import MultiHeadAttention


from models.func.clones import clones

class EncoderLayer(nn.Module):

    def __init__(self,size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()

        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x,mask):
    
        
        # we need to pass a func layer to sublayer as the 2nd parameter NOT the computational result
        # so we use lambda func here to map one input x to x,x,x 
        #print(mask.size())
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) 


        return self.sublayer[1](x, self.feed_forward)