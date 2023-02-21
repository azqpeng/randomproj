import os
import numpy as np
import torch
from torch import nn

"""
This class creates a neural network with variable depth and width.
"""
class mlp(nn.Module):
    # constructor
    def __init__(self, input_size=10, hidden_size=10, depth=2):
        super(mlp, self).__init__()
        self.flatten = nn.Flatten()
        # create linear blocks
        linear_blocks = [layer for i in range(depth) for layer in (nn.Linear(hidden_size, hidden_size), nn.ReLU())] 
        # stack
        self.stack = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), *linear_blocks, nn.Linear(hidden_size, 1), nn.ReLU())
    
    # Traditional forward pass :D 
    def forward(self, input_data):
        #input_data = self.flatten(input_data)
        logits = self.stack(input_data)
        return logits

    # This is mask pruning method
    def maskInit(self, subDim):
        # Ignores the input and output layers
        for i in range(1, len(self.stack) - 2):
            if isinstance(self.stack[i], nn.Linear):
                # get the weight matrix
                weight = self.stack[i].weight.data
                # get the weight matrix size
                weightSize = weight.size()
                # get the number of weights
                numWeights = weightSize[0]*weightSize[1]
                # get the number of weights to prune
                numPrune = int(numWeights - subDim)
                # get the indices of the weights to prune
                pruneIndices = np.random.choice(numWeights, numPrune, replace=False)
                # get the weight matrix as a 1D array
                weight1D = weight.view(-1)
                # prune the weights
                weight1D[pruneIndices] = 0
                # put the weights back into the weight matrix
                weight = weight1D.view(weightSize)
                # put the weights back into the layer
                self.stack[i].weight.data = weight


    # This is the random projection pruning method
    def projInit(self, subDim):
        # Ignores the input and output layers
        for i in range(1, len(self.stack) - 2):
            if isinstance(self.stack[i], nn.Linear):

                # get the weight matrix
                weight = self.stack[i].weight.data
                # get the weight matrix size
                weightSize = weight.size()
                # get the number of weights
                numWeights = weightSize[0]*weightSize[1]



