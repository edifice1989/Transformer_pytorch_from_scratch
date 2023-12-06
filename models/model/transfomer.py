import torch
import copy
import torch.nn as nn
from models.model import encode_decode
from models.model import encoder
from models.model import decoder

from models.layers import encoder_layer
from models.layers import decoder_layer

from models.layers import multi_head_attention
from models.layers import feed_forward_layer
from models.layers import positional_encoding

from models.func import embeddings
from models.func import generator
def make_model(
         src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    c = copy.deepcopy

    pos = positional_encoding(d_model) 

    attn = multi_head_attention(h, d_model)

    ff = feed_forward_layer(d_model, d_ff)

    model = encode_decode(
        encoder(encoder_layer(d_model, c(attn), c(ff), dropout), N),
        decoder(decoder_layer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(embeddings(d_model, src_vocab), c(pos)),#src embedded
        nn.Sequential(embeddings(d_model, tgt_vocab), c(pos)), #target embedded
        generator(d_model, tgt_vocab),
    )

       




