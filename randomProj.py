import os
import numpy as np
import torch
from torch import nn

"""
This class implements the functional that applies the random projection matrix and calculates it's gradient. 
"""

class randomProj(nn.Module()):
    def __init__(self, subRatio = 0.25, W_shape = (10, 10), weight = None) -> None:
        super(randomProj, self).__init__()

        # assign stuff
        self.W_shape = W_shape
        self.weight = weight

        # computes the dimension of the smaller subspace
        self.subSpace = int(subRatio * np.prod(W_shape))

        # initialize the subspace matrix weights.
        self.weight = nn.Parameter(torch.zeros(self.subSpace, dtype = torch.float), requires_grad = True)
        # TODO: Decide on how to actually make the initialization scale and if it should be uniform
        nn.init.uniform_(weight.data, a=-1, b=1)


        # create the random projection matrix that transforms the smaller subspace into the bigger one
        # TODO: Verify with Aditya that the std calculation is correct

        self.IDX = torch.normal(mean = 0, std = 1, size = (np.prod(W_shape), self.subSpace))

    # forward pass
    def forward(self):
        W = torch.matmul(self.IDX, self.weight)
        W = W.reshape(self.W_shape)
        return W

    # backpropogation steps, accounting for the random projection matrix
    
    # I have no clue what is happening here and I don't think it works
    def grad_small_to_large(self, grad):
        grad = torch.matmul(self.IDX, grad)
        return grad.reshape(self.W_shape)
    
    # Do I just implement this manually?
    def grad_large_to_small(self, grad):
        out_grad = torch.zeros(self.subSpace, dtype = torch.float)
        grad = grad.reshape(np.prod(self.W_shape), 1)
        for k in range(self.subSpace):
            for i in range(self.W_shape[0]):
                for j in range(self.W_shape[1]):
                    out_grad[k] +=  self.IDX[i * self.W_shape[1] + j, k]*grad[i * self.W_shape[1] + j]
        return out_grad
    


