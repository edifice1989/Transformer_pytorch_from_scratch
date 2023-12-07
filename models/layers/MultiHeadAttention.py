import torch
import torch.nn as nn
from models.func.self_attention import self_attention
from models.func.clones import clones

class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model,dropout=0.1):

        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
  # reduced dim for each Q,K,V, but added up to d_model
        self.d_k = d_model // h 
        
        self.h = h

        self.attn = None

  # 3 for K,Q,V, the forth layer is on the top for final attention score
        self.linears = clones(nn.Linear(d_model, d_model), 4) 

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, q, k, v, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches  = q.size(0) #q init as 512x512

    # split tensor by number of heads
      
        q, k, v = [   lin(x).view(nbatches , -1, self.h, self.d_k).transpose(1, 2)
    # [512,512] => [512,1,8,64] => [512,8,1,64] now we have 8 heads, 
    #length 1 since conv of size 1, dim of 64 for each q,k,v, 
    #ready for input to attention [batch_size, head, length, d_tensor]
            for lin, x in zip(self.linears, (q, k, v)) 
    # we only used 3 first linear layers since zip would 
        ]
        
    # calculate the attention score 
        x, self.attn = self_attention(q, k, v, mask=mask, dropout=self.dropout)

    # concat by view func [512, 8, 1, 64] => [512,1,512] add it back to 512
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del q
        del k
        del v

    # now apply the final linear layer copy
        return self.linears[-1](x) 
   