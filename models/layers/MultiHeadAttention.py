import torch
import torch.nn as nn
from models.func.attention import attention
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

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        #print (mask.size())
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)
   