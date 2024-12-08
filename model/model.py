import torch
import torch.nn as nn
import torch.nn.functional as F


class SoundPro(nn.Module):
    def __init__(self):
        super(SoundPro, self).__init__()
        
    
    def forward(self, x):
        
        return x

class VideoPro(nn.Module):
    def __init__(self):
        super(VideoPro, self).__init__()
        
    
    def forward(self, x):
        
        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        
    
    def forward(self, x):
        
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = Layers()  # Include the layers class
    
    def forward(self, x):
        return self.layers(x)  # Pass the input through the layers class
