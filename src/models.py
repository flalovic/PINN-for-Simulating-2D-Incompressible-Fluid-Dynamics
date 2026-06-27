import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_size, 256),
            nn.Tanh(),

            nn.Linear(256, 256),
            nn.Tanh(),

            nn.Linear(256, 256),
            nn.Tanh(),

            nn.Linear(256, 256),
            nn.Tanh(),

            nn.Linear(256, out_size),
        )

    def forward(self, x):
        return self.layers(x)