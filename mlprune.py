import os
import numpy as np
import torch
from torch import nn

class mlprune(nn.Module):
    # constructor
    def __init__(self, 
                 input_size=10, 
                 hidden_size=10, 
                 depth=2,
              compression = 1):
        super(mlprune, self).__init__()

        self.wsize = int(hidden_size**2 * compression)
        # create linear blocks
        linear_blocks = [layer for i in range(depth) for layer in (nn.Linear(hidden_size, hidden_size), nn.ReLU())] 
        # stack
        self.stack = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), *linear_blocks, nn.Linear(hidden_size, 1), nn.ReLU())
    
    # Traditional forward pass :D 
    def forward(self, input_data):
        #input_data = self.flatten(input_data)
        logits = self.stack(input_data)
        return logits
    

    