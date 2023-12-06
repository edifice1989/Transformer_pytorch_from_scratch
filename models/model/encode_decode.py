import torch
import torch.nn as nn



class encode_decode(nn.Module):

    def __init__(self,encoder,decoder,src_embed, tgt_embed, generator):

        super(encode_decode).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = tgt_embed
        self.generator = generator

    def forward(self,src,src_mask,target, target_mask):

        return self.decode(self.encode(src,src_mask),src_mask,target, target_mask)
    
    def encode(self,src,src_mask):

        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory,src_mask,target, target_mask):

        return self.decoder(self.target_embed(target),memory,src_mask,target_mask)


