import math
import torch
#attention 
def self_attention(k,q,v,mask=None, dropout=None):
    # q dim [batch_size,n_heads,length,d_tensor]

    d_tensor = q.size(-1) 

    # assume dim of query/key/value vector should be same 
    # and it should be to make below calculation happen      
    k_t = k.transpose(-2,-1) #[batch_size,n_heads,d_tensor,length]

    score = (q @ k_t)/math.sqrt(d_tensor)
    if mask is not None:
        score = score.masked_fill_(mask == 0, -1e9)

    p_attn = score.softmax(dim=-1)

    if dropout is not None:

        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, v), p_attn