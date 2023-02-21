# This is the model trainer

import os
import numpy as np
import torch
from torch import nn

# is the parameter space input_size x depth?

hidden_size = 10
depth = 3

parameter_space = hidden_size*depth

subspace = parameter_space*0.7