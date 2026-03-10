import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from typing import Optional

class BaseNN(nn.Module):
    def __init__(self, input_dim, output_dim, name: str = "model", device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.device = torch.device(device)
        self.dtype = dtype

        self.loss: list[float] = []
        self.val_loss: list[float] = []
        self.training_time: Optional[float] = None
        self.epochs_trained: int = 0
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass