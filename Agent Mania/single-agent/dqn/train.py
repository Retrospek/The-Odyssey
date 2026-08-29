import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
 
from models.value_model import ValueBaseModel
from models.buffer import ERB
from models.dqn.dqn_trainer import DQN_trainer
from models.dqn.dqn import DQN
 
# ----------------------------------------------------------------------
# Q-network
# ----------------------------------------------------------------------

def build_qnet(obs_dim: int, action_dim: int, hidden: int = 128) -> DQN:
    net = nn.Sequential(
        nn.Linear(obs_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, action_dim),
    )
    q_net = DQN(net, tau=1.0)   # tau here just gets stored; trainer still passes its own tau at call time
    q_net.apply(q_net.init_weights)
    return q_net
  
# ----------------------------------------------------------------------
# Hyperparameter schedule functions (plain decay, tweak as needed)
# ----------------------------------------------------------------------
def linear_epsilon_update(epsilon, epsilon_min, epsilon_decay):
    return max(epsilon_min, epsilon - epsilon_decay)
 
 
def constant_update(value, decay):
    # tau and gamma held fixed; pass decay=0.0 if you want them static
    return value - decay
 
 
def lr_update(lr, decay):
    return lr  # no LR scheduling by default
 
 
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_name="LunarLander-v3"
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]      # 4
    action_dim = env.action_space.n                # 2
 
    policy_net = build_qnet(obs_dim, action_dim).to(device)
    target_net = build_qnet(obs_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
 
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
 
    buffer = ERB(n=10000)
 
    trainer = DQN_trainer(
        batch_size=32,
        episode_num=300,
        max_steps=500,                 
        update_step_interval=100,      
 
        tau=1.0,                       
        tau_decay=0.0,
        tau_update=constant_update,
 
        gamma=0.99,
        gamma_decay=0.0,
        gamma_update=constant_update,
 
        learning_rate=1e-3,
        learning_rate_decay=0.0,
        learning_rate_update=lr_update,
 
        buffer=buffer,
        buffer_feature=None,           
        action_dim=action_dim,
 
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995 / 100,     
        epsilon_update=linear_epsilon_update,
 
        policy_net=policy_net,
        target_net=target_net,
        optim=optimizer,
        criterion=criterion,
 
        device=device,
        dash_path=rf"Agent Mania\single-agent\dqn\{env_name}_dqn_training_dash",
        env=env,
    )
 
    trainer.train_step()
 
    # ------------------------------------------------------------------
    # Quick summary + checkpoint
    # ------------------------------------------------------------------
    rewards = trainer.episode_rewards
    last_100 = rewards[-100:]
    avg_last_100 = sum(last_100) / len(last_100)
    print(f"Finished {len(rewards)} episodes.")
    print(f"Avg reward (last 100 episodes): {avg_last_100:.1f}")
    print(f"Solved (>=475 avg over last 100)? {'YES' if avg_last_100 >= 475 else 'no'}")
 
    policy_net.save_checkpoint("checkpoints/cartpole_dqn.pt")
    print("Saved checkpoint to checkpoints/cartpole_dqn.pt")
 
    env.close()
 
 
if __name__ == "__main__":
    main()