import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import time
import torch
import torch.nn as nn
import gymnasium as gym

from models.dqn.dqn import DQN

NUM_TRIALS = 10
CHECKPOINT_PATH = "checkpoints/cartpole_ddqn.pt"


def build_qnet(obs_dim: int, action_dim: int, hidden: int = 128) -> DQN:
    net = nn.Sequential(
        nn.Linear(obs_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, action_dim),
    )
    return DQN(net, tau=1.0)


def main():
    device = "cpu"  # rendering + single-step inference — no need for GPU here

    env = gym.make("LunarLander-v3", render_mode="human")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = build_qnet(obs_dim, action_dim).to(device)
    policy_net.load_checkpoint(CHECKPOINT_PATH)
    policy_net.eval()

    episode_lengths = []

    for trial in range(NUM_TRIALS):
        state, _ = env.reset()
        done = False
        steps = 0

        while not done:
            state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(policy_net(state_t), dim=1).item()  # greedy, no epsilon

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            env.render()
            time.sleep(0.01)  # slow it down slightly so it's watchable

        episode_lengths.append(steps)
        print(f"Trial {trial + 1}: survived {steps} steps")

    print(f"\nAverage over {NUM_TRIALS} trials: {sum(episode_lengths) / NUM_TRIALS:.1f} steps")
    env.close()


if __name__ == "__main__":
    main()