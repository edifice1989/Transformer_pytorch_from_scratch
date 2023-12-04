import math
import torch
#attention 
def attention(k,q,v):
    # q dim [batch_size,n_heads,length,d_tensor]

    d_tensor = q.size(-1) 

    # assume dim of query/key/value vector should be same 
    # and it should be to make below calculation happen      
    k_t = k.transpose(-2,-1) #[batch_size,n_heads,d_tensor,length]

    score = (q @ k_t)/math.sqrt(d_tensor)

    v= torch.softmax(score,dim=-1) @ v

    return v,score