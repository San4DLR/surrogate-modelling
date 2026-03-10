import time
from typing import Optional 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from nnmodels.base import BaseNN

class FeedForward(BaseNN):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_layers: list[int] = [64, 64], 
                 hidden_activation: nn.Module = nn.ReLU(),
                 out_activation: Optional[nn.Module] = None,
                 **kwargs):
        
        super().__init__(input_dim, output_dim, **kwargs)

        self.first_layer = nn.Linear(input_dim, hidden_layers[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_layers[i], hidden_layers[i + 1]) for i in range(len(hidden_layers) - 1)])
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)

        self.hidden_activation = hidden_activation
        self.out_activation = out_activation
    
    def forward(self, input, grad=True):

        if not grad:
            with torch.no_grad():
                pass
        x = self.hidden_activation(self.first_layer(input))
        for hidden_layer in self.hidden_layers:
            x = self.hidden_activation(hidden_layer(x))

        if self.out_activation == None:
                output = self.output_layer(x)
        else:
            output = self.out_activation(self.output_layer(x))

        return output