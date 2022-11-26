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
        super(Network, self).__init__()
        self.input_size = input_size # 5 neuron in input layer of neural network
        self.nb.action = nb_action # 3 neuron in action layer of neural network
        self.fc1 = nn.Linear(input_size, 30) # to make connection between the input layer and the hidden layer
        # 30 is the neurons in the hidden layer, this value can be anything but after trying many architectures, this seems best
        self.fc2 = nn.Linear(30, nb_action)
        # Neural Network = 5 -> 30 -> 3
    
    # forward propagation
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values= self.fc2(x)
        return q_values
    

# Experience replays

class replayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]