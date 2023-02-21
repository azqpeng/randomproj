import os
import numpy as np
import torch
from torch import nn

"""
This class creates a neural network with variable depth and width.
"""
class mask(nn.Module):
    # create model
    def __init__(self, input_size=10, hidden_size=10, depth=2):
        super(mask, self).__init__()
        self.flatten = nn.Flatten()
        # create linear blocks
        linear_blocks = [nn.Linear(hidden_size, hidden_size) for _ in range(depth)] 
        # stack
        self.stack = nn.Sequential(nn.Linear(input_size, hidden_size), *linear_blocks, nn.Linear(hidden_size, 1))

        # initialize activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    # feed
    def forward(self, x):
        #input_data = self.flatten(input_data)
        x = self.stack(x)
        return x
