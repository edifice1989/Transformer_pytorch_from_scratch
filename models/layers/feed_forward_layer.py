import torch
import torch.nn as nn
class feed_forward_layer(nn.Module):

    def __init__(self, d_model, hidden):

        super(feed_forward_layer, self).__init__()

        self.linear1 = nn.Linear(d_model, hidden)

        self.linear2 = nn.Linear(hidden, d_model)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.linear1(x)

        x = self.relu(x)

        x = self.linear2(x)

        return x