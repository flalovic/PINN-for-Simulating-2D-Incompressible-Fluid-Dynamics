import torch
import torch.nn as nn

class PINN(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()

        self.fc1 = nn.Linear(in_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, out_size)
        
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)

        return x