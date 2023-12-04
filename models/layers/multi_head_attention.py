import torch
from models.layers import self_attention
from models.layers import clones

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):

        super(MultiHeadAttention, self).__init__()
  # reduced dim for each Q,K,V, but added up to d_model
        self.d_k = d_model // n_head 
        
        self.n_head = n_head

        self.attn = None

  # use the attention class defined above
        self.attention = self_attention() 

  # 3 for K,Q,V, the forth layer is on the top for final attention score
        self.linears = clones(nn.Linear(d_model, d_model), 4) 



    def forward(self, q, k, v):
        samples = q.size(0) #q init as 512x512

    # split tensor by number of heads
      
        q, k, v = [   lin(x).view(samples, -1, self.n_head, self.d_k).transpose(1, 2)
    # [512,512] => [512,1,8,64] => [512,8,1,64] now we have 8 heads, 
    #length 1 since conv of size 1, dim of 64 for each q,k,v, 
    #ready for input to attention [batch_size, head, length, d_tensor]
            for lin, x in zip(self.linears, (q, k, v)) 
    # we only used 3 first linear layers since zip would 
        ]
        
    # calculate the attention score 
        x, self.attn = attention(q, k, v)

    # concat by view func [512, 8, 1, 64] => [512,1,512] add it back to 512
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(samples, -1, self.n_head * self.d_k)
        )

    # now apply the final linear layer copy
        return self.linears[-1](x) 
   