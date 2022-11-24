# Importing the libraries

import numpy as np
import random
import os # to load and save the model
import torch # For Neural Network
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Creating the architecture of the Neural Network 

# the class Network is a child class and its parent class is nn.Module
class Network(nn.Module):
    def __init__(self, input_size, nb_action):
        # We will pass the input size and output size in self, input size is 5 (All 3 signals, orientation, etc) and action or output is of size 3 (Straight, left, right)