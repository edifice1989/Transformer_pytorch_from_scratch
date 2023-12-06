import torch
import torch.nn as nn

from models.model import Transfomer

lr= 0.01

model = transformer()
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
loss_func = 
epoch = 1000

for e in range(epoch):
    model.train()
    loss = model

