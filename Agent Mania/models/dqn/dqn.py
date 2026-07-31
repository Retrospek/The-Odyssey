from models.model import BaseModel
import torch
import torch.nn as nn

class DQN(BaseModel):
    def __init__(self, network:nn.Module, tau):
        super().__init__()

        self.q_head = network

    def forward(self, x):
        return self.q_head(x)