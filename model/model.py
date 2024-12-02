import torch
import torch.nn as nn
import torch.nn.functional as F


class Layers(nn.Module):
    def __init__(self):
        super(Layers, self).__init__()
        
    
    def forward(self, x):
        
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = Layers()  # Include the layers class
    
    def forward(self, x):
        return self.layers(x)  # Pass the input through the layers class
