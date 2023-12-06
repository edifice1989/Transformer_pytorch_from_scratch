import torch
import torch.nn as nn
from models.block import encoder_block
from models.block import decoder_block
from models.model import transfomer

lr= 0.01

model = transformer()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
loss_func = 
epoch = 1000

for e in range(epoch):
    model.train()
    loss = model

