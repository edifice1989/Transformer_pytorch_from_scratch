import torch
import torch.nn as nn
class FeedForwardLayer(nn.Module):

    def __init__(self, d_model, hidden,dropout=0.1):

        super(FeedForwardLayer, self).__init__()

        self.linear1 = nn.Linear(d_model, hidden)

        self.linear2 = nn.Linear(hidden, d_model)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.linear1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.linear2(x)

        return x