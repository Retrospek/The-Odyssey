import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pg_model import PGBaseModel

class TRPO(PGBaseModel):
    def __init__(self, trunk_net: nn.Module, value_head: nn.Module, policy_net: nn.Module) -> None:
        super().__init__()
        self.trunk_net = trunk_net
        self.value_head = value_head
        self.policy_net = policy_net

    def forward(self, x: torch.Tensor) -> dict:
        """
        Intuition:
        - This is a actor-critic child class version of the inherited PGBaseModel
        - 
        """
        features = self.trunk_net(x)
        state_value = self.value_head(features)
        action_probs = self.policy_net(features)
        log_action_probs = torch.log(action_probs)

        return {
            "state_value": state_value.squeeze(),
            "action_probs": action_probs.squeeze(0),
            "log_action_probs": log_action_probs.squeeze(0),
        }