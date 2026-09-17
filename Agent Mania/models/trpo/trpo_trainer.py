from models.buffer import BaseBuffer
from models.experience import make_transition_class
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical

from collections.abc import Callable

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

Transition = make_transition_class("state next_state action reward old_log_prob state_val done")

class TRPO_trainer:
    def __init__(self, batch_size: int, episode_num: int, max_steps: int,
                 gamma: float, gamma_decay: float, gamma_update: Callable,
                 gae_lambda: float,
                 max_kl: float, damping_coeff: float, cg_iters: int, cg_residual_tol: float,
                 backtrack_iters: int, backtrack_coeff: float,
                 value_lr: float, value_train_iters: int,
                 buffer: BaseBuffer, buffer_feature: str, action_dim: int, state_dim: int,
                 actor_critic: nn.Module,
                 device: str, dash_path: str, env) -> None:

        self.batch_size = batch_size
        self.episode_num = episode_num
        self.max_steps = max_steps

        self.gamma = gamma
        self.gamma_decay = gamma_decay
        self.gamma_update = gamma_update
        self.gae_lambda = gae_lambda

        self.max_kl = max_kl
        self.damping_coeff = damping_coeff
        self.cg_iters = cg_iters
        self.cg_residual_tol = cg_residual_tol
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff

        self.value_lr = value_lr
        self.value_train_iters = value_train_iters
        # Adjust this to however your actor_critic exposes the value head's
        # parameters (e.g. self.actor_crsitic.value_net.parameters()).
        self.value_optim = torch.optim.Adam(self.actor_critic.parameters(), lr=value_lr)

        self.buffer = buffer
        self.buffer_feature = buffer_feature
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.actor_critic = actor_critic
        self.device = device
        self.dash_path = dash_path
        self.env = env

        # ---- logging ----
        self.episode_rewards = []
        self.episode_steps = []
        self.policy_loss_history = []   # surrogate objective, per update
        self.value_loss_history = []    # value MSE, per update
        self.kl_history = []            # realized KL after line search, per update
        self.gamma_history = []

    # -----------------------------------------------------------------
    # Forward pass helper (kept from your original)
    # -----------------------------------------------------------------
    def output_stateval_and_probs(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.actor_critic(state_t)

    # -----------------------------------------------------------------
    # 1. Rollout collection — replaces off-policy buffer sampling
    # -----------------------------------------------------------------
    def collect_rollout(self):
        states, next_states, actions = [], [], []
        rewards, old_log_probs, values, dones = [], [], [], []

        state, _ = self.env.reset()
        episode_reward = 0.0
        episode_steps = 0

        for b in range(self.batch_size):
            stateval_and_probs = self.output_stateval_and_probs(state)
            state_value = stateval_and_probs["state_value"]
            action_probs = stateval_and_probs["action_probs"]
            log_action_probs = stateval_and_probs["log_action_probs"]

            dist = Categorical(probs=action_probs)
            action = dist.sample()
            log_prob = log_action_probs[action]

            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            states.append(state)
            next_states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            old_log_probs.append(log_prob)
            values.append(state_value)
            dones.append(done)

            episode_reward += reward
            episode_steps += 1
            state = next_state

            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_steps.append(episode_steps)
                self.gamma_history.append(self.gamma)
                episode_reward = 0.0
                episode_steps = 0
                state, _ = self.env.reset()

        if dones[-1]:
            last_value = 0.0
        else:
            last_value = self.output_stateval_and_probs(next_states[-1])["state_value"].item()

        return {
            "states": torch.as_tensor(np.array(states), dtype=torch.float32, device=self.device),
            "next_states": torch.as_tensor(np.array(next_states), dtype=torch.float32, device=self.device),
            "actions": torch.stack(actions).to(self.device),
            "rewards": torch.as_tensor(rewards, dtype=torch.float32, device=self.device),
            "old_log_probs": torch.stack(old_log_probs).to(self.device),
            "values": torch.stack(values).to(self.device),
            "dones": torch.as_tensor(dones, dtype=torch.float32, device=self.device),
            "last_value": last_value,
        }


    # -----------------------------------------------------------------
    # 2. Advantage estimation (GAE)
    # -----------------------------------------------------------------
    def compute_gae(self, rewards, values, dones, last_value):
        advantages=[-1] * len(rewards)

        for t in range(len(rewards) - 1, -1, -1):
            r_t=rewards[t]
            v_t=values[t]
            d_t=dones[t]

            if t == len(rewards) - 1:
                v_tpo = last_value
                next_advantage = 0.0
            else:
                v_tpo = values[t + 1]
                next_advantage = advantages[t + 1]

            delta = r_t + self.gamma * v_tpo * (1 - d_t) - v_t
            advantage = delta + self.   gamma * self.gae_lambda * (1 - d_t) * next_advantage
            advantages[t] = advantage

        advantages=torch.tensor(advantages, dtype=torch.float32)
        returns=advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        
        return {
            "advantages": advantages,
            "returns": returns
        }

    # -----------------------------------------------------------------
    # 3. Surrogate objective
    # -----------------------------------------------------------------
    def surrogate_loss(self, states, actions, old_log_probs, advantages):
        logits = self.actor_critic(states)
        log_action_probs = logits["log_action_probs"]  # (batch_size, action_dim)

        selected_log_probs = log_action_probs.gather(1, actions.unsqueeze(1)).squeeze(1)  # (batch_size,)

        ratios = torch.exp(selected_log_probs - old_log_probs)
        return torch.mean(ratios * advantages)

    # -----------------------------------------------------------------
    # 4. Mean KL divergence between old and current policy
    # -----------------------------------------------------------------
    def mean_kl(self, states):
        logits=self.actor_critic(states)    
        log_probs=logits["log_action_probs"]

        old_log_probs=log_probs.detach() # log(P(a)) remember we don't want this to be 
        new_log_probs=log_probs # log(Q(a))

        probs=logits["action_probs"]
        old_probs=probs.detach()
        new_probs=probs

        per_state_kl = torch.sum(old_probs * (old_log_probs - new_log_probs), dim=1)
        mean_kl = torch.mean(per_state_kl)
        return mean_kl

    # -----------------------------------------------------------------
    # 5. Fisher-vector product — never forms the full Fisher matrix
    # -----------------------------------------------------------------
    def fisher_vector_product(self, states, vector):
        raise NotImplementedError

    # -----------------------------------------------------------------
    # 6. Conjugate gradient solve: F x = g
    # -----------------------------------------------------------------
    def conjugate_gradient(self, states, g):
        raise NotImplementedError

    # -----------------------------------------------------------------
    # 7. Step size + backtracking line search
    # -----------------------------------------------------------------
    def line_search(self, states, actions, old_log_probs, advantages, x):
        raise NotImplementedError

    # -----------------------------------------------------------------
    # 8. Policy update — ties 3 through 7 together. No optimizer.step()
    #    anywhere in this method — the update is applied manually.
    # -----------------------------------------------------------------
    def update_policy(self, states, actions, old_log_probs, advantages):
        raise NotImplementedError

    # -----------------------------------------------------------------
    # 9. Value function update — ordinary supervised regression.
    #    This is the one place a normal optimizer.step() belongs.
    # -----------------------------------------------------------------
    def update_value(self, states, returns):
        raise NotImplementedError

    # -----------------------------------------------------------------
    # 10. Main loop
    # -----------------------------------------------------------------
    def train_step(self):
        raise NotImplementedError

    # -----------------------------------------------------------------
    # Dashboard — kept and relabeled for TRPO. Epsilon/tau panels
    # dropped (not applicable); KL and value-loss panels added instead.
    # -----------------------------------------------------------------
    def plot_dashboard(self, save_path: str = None, show: bool = True, ma_window: int = 20):
        if len(self.episode_rewards) == 0:
            raise ValueError("No training metrics recorded yet — run train_step() first.")

        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['figure.facecolor'] = 'white'

        episodes = np.arange(1, len(self.episode_rewards) + 1)
        rewards = np.array(self.episode_rewards, dtype=float)

        def moving_average(x, window):
            window = max(1, min(window, len(x)))
            return np.convolve(x, np.ones(window) / window, mode='valid')

        colors = {
            'reward': '#3498db', 'reward_ma': '#e74c3c',
            'policy_loss': '#9b59b6', 'value_loss': '#e67e22',
            'kl': '#1abc9c', 'gamma': '#f39c12', 'steps': '#95a5a6',
        }

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)
        fig.suptitle("TRPO Training Dashboard", fontsize=20, fontweight='bold', color='#2c3e50', y=0.98)

        def style_axis(ax):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(alpha=0.2)

        # 1. Reward per episode + moving average
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(episodes, rewards, color=colors['reward'], alpha=0.35, linewidth=1, label='Episode Reward')
        if len(rewards) >= 2:
            ma = moving_average(rewards, ma_window)
            ax1.plot(episodes[-len(ma):], ma, color=colors['reward_ma'], linewidth=2.5, label=f'Moving Avg ({ma_window})')
        ax1.set_title("Episode Reward", fontsize=13, fontweight='bold', loc='left')
        ax1.set_xlabel("Episode"); ax1.set_ylabel("Reward")
        ax1.legend(frameon=False); style_axis(ax1)

        # 2. Policy (surrogate) loss per update
        ax2 = fig.add_subplot(gs[1, 0])
        if len(self.policy_loss_history) > 0:
            pl = np.array(self.policy_loss_history, dtype=float)
            ax2.plot(np.arange(1, len(pl) + 1), pl, color=colors['policy_loss'], linewidth=1.5)
        ax2.set_title("Surrogate Objective", fontsize=13, fontweight='bold', loc='left')
        ax2.set_xlabel("Update"); ax2.set_ylabel("Value")
        style_axis(ax2)

        # 3. Value loss per update
        ax3 = fig.add_subplot(gs[1, 1])
        if len(self.value_loss_history) > 0:
            vl = np.array(self.value_loss_history, dtype=float)
            ax3.plot(np.arange(1, len(vl) + 1), vl, color=colors['value_loss'], linewidth=1.5)
        ax3.set_title("Value Loss (MSE)", fontsize=13, fontweight='bold', loc='left')
        ax3.set_xlabel("Update"); ax3.set_ylabel("Loss")
        style_axis(ax3)

        # 4. Realized KL per update (should hover under max_kl)
        ax4 = fig.add_subplot(gs[2, 0])
        if len(self.kl_history) > 0:
            klh = np.array(self.kl_history, dtype=float)
            ax4.plot(np.arange(1, len(klh) + 1), klh, color=colors['kl'], linewidth=1.5, label='Realized KL')
            ax4.axhline(self.max_kl, color='#e74c3c', linestyle='--', linewidth=1.5, label='max_kl')
        ax4.set_title("KL Divergence per Update", fontsize=13, fontweight='bold', loc='left')
        ax4.set_xlabel("Update"); ax4.set_ylabel("KL")
        ax4.legend(frameon=False); style_axis(ax4)

        # 5. Episode length + gamma decay
        ax5 = fig.add_subplot(gs[2, 1])
        ax5b = ax5.twinx()
        l1, = ax5.plot(episodes, self.episode_steps, color=colors['steps'], linewidth=1.5, label='Steps/Episode')
        l2, = ax5b.plot(episodes, self.gamma_history, color=colors['gamma'], linewidth=2, label='Gamma')
        ax5.set_title("Episode Length & Gamma", fontsize=13, fontweight='bold', loc='left')
        ax5.set_xlabel("Episode")
        ax5.set_ylabel("Steps", color=colors['steps'])
        ax5b.set_ylabel("Gamma", color=colors['gamma'])
        ax5.tick_params(axis='y', labelcolor=colors['steps'])
        ax5b.tick_params(axis='y', labelcolor=colors['gamma'])
        ax5b.spines['top'].set_visible(False)
        style_axis(ax5)
        ax5.legend(handles=[l1, l2], labels=['Steps/Episode', 'Gamma'], frameon=False, loc='upper right')

        last_n = min(20, len(rewards))
        fig.text(
            0.5, 0.01,
            f"Episodes: {len(episodes)}   |   Best Reward: {rewards.max():.1f}   |   "
            f"Avg Reward (last {last_n}): {rewards[-last_n:].mean():.1f}",
            ha='center', fontsize=11, color='#7f8c8d'
        )

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        if show:
            plt.show()

        return fig