import os
import numpy as np
import torch
from torch import nn

"""
This class creates a neural network with variable depth and width.
"""
class mlp(nn.Module):
    # constructor
    def __init__(self, pruneType = None, input_size=10, hidden_size=10, depth=2, activation = nn.ReLU(), learning_rate = 0.01, optimizer = torch.optim.Adam, compression = 0.25):
        super(mlp, self).__init__()
            
        # create module list
        self.layers = nn.ModuleList()

        # add input layer
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(activation)

        # add hidden layers
        for i in range(depth - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(activation)
        
        # add output layer
        self.layers.append(nn.Linear(hidden_size, 1))
        self.layers.append(nn.Sigmoid())

        # all that stuff to make sure the network is on the GPU and training right
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.learning_rate = learning_rate
        self.optimizer = optimizer(params=self.parameters(), lr=learning_rate)

        # check type of model to make
        if pruneType == 'mask':
            self.maskInit(compression)
            print("Mask pruned model created.")
        elif pruneType == 'randomProj':
            self.randomProj(compression)
            print("Random projection model created.")
        else:
            print("Non-pruned model created.")


    # Traditional forward pass :D 
    def forward(self, input_data):
        # loop through layers
        logits = input_data
        for layer in self.layers:
            logits = layer(logits)
        return logits

    # This is mask pruning method
    def maskInit(self, compression):
        for i in range(1, len(self.layers) - 1):
            if isinstance(self.layers[i], nn.Linear):
                # get the weight matrix
                weight = self.layers[i].weight.data
                # get the weight matrix size
                weightSize = weight.size()
                # get the number of weights
                numWeights = weightSize[0]*weightSize[1]
                # get the number of weights to prune
                numPrune = int(numWeights * (1 - compression))
                # get the indices of the weights to prune
                pruneIndices = np.random.choice(numWeights, numPrune, replace=False)
                # get the weight matrix as a 1D array
                weight1D = weight.view(-1)
                # prune the weights
                weight1D[pruneIndices] = 0
                # put the weights back into the weight matrix
                weight = weight1D.view(weightSize)
                # put the weights back into the layer
                self.layers[i].weight.data = weight
    
    # This is the random gaussian projection model
    def randomProj(self, compression):
        for i in range(1, len(self.layers) - 1):
            if isinstance(self.layers[i], nn.Linear):
                # get the weight matrix
                weight = self.layers[i].weight.data
                # get the weight matrix size
                weightSize = weight.size()
                # get the number of weights
                numWeights = weightSize[0]*weightSize[1]
                # get the number of weights to prune
                numSS = int(numWeights * compression)
                """
                PSEUDOCODE: 
                Create a vector of size numSS

                
                
                
                
                """




                





