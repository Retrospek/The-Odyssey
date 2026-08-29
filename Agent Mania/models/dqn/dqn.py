from models.value_model import ValueBaseModel
import torch.nn as nn

class DQN(ValueBaseModel):
    def __init__(self, network:nn.Module):
        super().__init__()

        self.q_head = network

    def forward(self, x):
        return self.q_head(x)