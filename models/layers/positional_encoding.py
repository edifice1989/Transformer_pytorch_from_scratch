import torch
import torch.nn as nn


class positional_encoding(nn.module):

    def __init__(self,d_model,max_len):

        super(positional_encoding).__init__()

        self.encoding = torch.zeros(max_len,d_model) # figure out the order of those two, 
        # how to use them in the below section

        pos = torch.range(0,max_len).float()

        pos = pos.unsqueeze(dim=1)

        _i2 = torch.arrange(0,d_model,step=2).float()

        self.encoding[:,0::2] = torch.sin(pos/10000**(_i2/d_model))

        self.encoding[:,1::2] = torch.cos(pos/10000**(_i2/d_model))

    def forward(self,x):
        
        batch_size, seq_len = x.size() 

        return self.encoding[:seq_len, :]
