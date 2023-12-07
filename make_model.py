import torch
import copy
import torch.nn as nn


from models.model.EncodeDecode import EncodeDecode
from models.model.Encoder import Encoder
from models.model.Decoder import Decoder

from models.layers.EncoderLayer import EncoderLayer
from models.layers.DecoderLayer import DecoderLayer

from models.layers.PositionalEncoding import PositionalEncoding
from models.layers.MultiHeadAttention import MultiHeadAttention
from models.layers.FeedForwardLayer import FeedForwardLayer


from models.func.Embeddings import Embeddings
from models.func.Generator import Generator
def make_model(
         src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    c = copy.deepcopy

    pos = PositionalEncoding(d_model,dropout) 

    attn = MultiHeadAttention(h, d_model)

    ff = FeedForwardLayer(d_model, d_ff)

    model = EncodeDecode(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(pos)),#src embedded
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(pos)), #target embedded
        Generator(d_model, tgt_vocab),
    )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.kaiming_uniform_(p)
    return model

       




