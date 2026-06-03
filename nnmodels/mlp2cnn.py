import torch
from torch import tensor
import torch.nn as nn

import torch
import torch.nn as nn

class MLP2CNN(nn.Module):
    def __init__(
        self,
        input_dim,          # size of geometry input vector
        mlp_hidden_dim,     # hidden dim of MLP layers
        mlp_num_layers,     # number of MLP hidden layers
        mlp_activation,     # e.g. nn.ReLU(), nn.Tanh(), nn.GELU()
        cnn_channels,       # list of channel sizes e.g. [1, 16, 8, 1]
        cnn_kernel_size,    # e.g. 3
        output_size,        # spatial output e.g. 128
    ):
        super().__init__()

        self.output_size = output_size
        self.start_channels = cnn_channels[0]  # first CNN channel count

        # --- MLP ---
        mlp_layers = []
        in_dim = input_dim
        for _ in range(mlp_num_layers):
            mlp_layers.append(nn.Linear(in_dim, mlp_hidden_dim))
            mlp_layers.append(mlp_activation)
            in_dim = mlp_hidden_dim
        # final MLP layer → flatten spatial feature map
        mlp_layers.append(nn.Linear(in_dim, cnn_channels[0] * output_size * output_size))
        mlp_layers.append(mlp_activation)
        self.mlp = nn.Sequential(*mlp_layers)

        # --- CNN ---
        cnn_layers = []
        padding = cnn_kernel_size // 2  # keeps spatial size unchanged
        for i in range(len(cnn_channels) - 1):
            cnn_layers.append(nn.Conv2d(cnn_channels[i], cnn_channels[i+1],
                                        kernel_size=cnn_kernel_size,
                                        padding=padding))
            if i < len(cnn_channels) - 2:  # no activation after last layer
                cnn_layers.append(mlp_activation)
        self.cnn = nn.Sequential(*cnn_layers)

    def forward(self, x):
        x = self.mlp(x)                                          # (batch, C * H * W)
        x = x.view(-1, self.start_channels, self.output_size, self.output_size)  # (batch, C, H, W)
        x = self.cnn(x)                                          # (batch, 1, H, W)
        return x