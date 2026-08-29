import os
import torch
import torch.nn as nn

class PGBaseModel(nn.Module):

  def __init__(self, **kwargs) -> None:
    super().__init__()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Every child class must implement its forward pass."""
    raise NotImplementedError("Child classes must implement forward().")

  def init_weights(self, network: nn.Module) -> None:
    """
    Standardized weight initialization helper used for RL algos because we don't want bias so that an action ends up getting over sampled (0-bias)
    We also want there to be no 
    """
    if isinstance(network, (nn.Linear, nn.Conv2d)):
      nn.init.orthogonal_(network.weight, gain=nn.init.calculate_gain('relu'))
      if network.bias is not None:
        nn.init.zeros_(network.bias)

  def soft_update(self, target_model: nn.Module, tau: float) -> None:
    """Soft updates target network weights: theta_target = tau*theta_local + (1 - tau)*theta_target

    Essential for DQN, DDQN, and Off-Policy Actor-Critic models.
    """
    for target_param, local_param in zip(target_model.parameters(), self.parameters()):
      target_param.data.copy_(
          tau * local_param.data + (1.0 - tau) * target_param.data
      )

  def save_checkpoint(self, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(self.state_dict(), filepath)

  def load_checkpoint(self, filepath: str) -> None:
    if not os.path.isfile(filepath):
      raise FileNotFoundError(f"No checkpoint found at {filepath}")
    self.load_state_dict(torch.load(filepath))