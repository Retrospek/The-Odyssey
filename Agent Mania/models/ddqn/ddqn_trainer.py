from models.buffer import BaseBuffer
from models.experience import make_transition_class
from models.dqn.dqn import DQN
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from collections.abc import Callable

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

Transition = make_transition_class()  # build the namedtuple class once


class DDQN_trainer:
    def __init__(self, batch_size:int, episode_num:int, max_steps:int, update_step_interval:int,
                 tau:float, tau_decay:float, tau_update:Callable,
                 gamma:float, gamma_decay:float, gamma_update:Callable,
                 learning_rate:float, learning_rate_decay:float, learning_rate_update:Callable,
                 buffer:BaseBuffer, buffer_feature:str, action_dim:int,
                 epsilon:float, epsilon_min:float, epsilon_decay:float, epsilon_update:Callable,
                 action_net:nn.Module, q_net:nn.Module, optim:optim.Optimizer, criterion:nn.modules.loss._Loss,
                 device:str, dash_path:str, env) -> None:

        # --------------------------------
        self.episode_num=episode_num
        self.max_steps=max_steps
        self.update_step_interval=update_step_interval
        self.batch_size=batch_size

        self.tau=tau
        self.tau_decay=tau_decay
        self.tau_update=tau_update

        self.gamma=gamma
        self.gamma_decay=gamma_decay
        self.gamma_update=gamma_update

        self.LR=learning_rate
        self.LR_decay=learning_rate_decay
        self.LR_update=learning_rate_update

        self.buffer=buffer
        self.buffer_feature=buffer_feature
        self.action_dim=action_dim

        self.epsilon=epsilon
        self.epsilon_min=epsilon_min
        self.epsilon_decay=epsilon_decay
        self.epsilon_update=epsilon_update

        self.action_net=action_net
        self.q_net=q_net
        self.optim=optim
        self.criterion=criterion

        self.device=device
        self.dash_path=dash_path
        self.env=env
        # --------------------------------
        self.episode_losses=[]
        self.episode_rewards=[]
        self.episode_steps=[]
        self.epsilon_history=[]
        self.gamma_history=[]
        self.tau_history=[]
        self.lr_history=[]
        self._step_counter = 0  
        # --------------------------------

    def select_action(self, state) -> int:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        rand_val = torch.rand(1).item()
        with torch.no_grad():
            if rand_val <= self.epsilon:
                action = torch.randint(0, self.action_dim, (1,)).item()  
            else:
                action = torch.argmax(self.action_net(state_t), dim=1).item()  

        return action

    def update_net(self):
        if len(self.buffer) < self.batch_size:
            return None

        experience_samples = self.buffer.pop(self.batch_size)

        states = torch.as_tensor(np.array([item.state for item in experience_samples]), dtype=torch.float32, device=self.device)
        next_state = torch.as_tensor(np.array([item.next_state for item in experience_samples]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([item.action for item in experience_samples], dtype=torch.long, device=self.device).unsqueeze(1)  
        rewards = torch.tensor([item.reward for item in experience_samples], dtype=torch.float32, device=self.device).unsqueeze(1)  
        dones = torch.tensor([item.done for item in experience_samples], dtype=torch.float32, device=self.device).unsqueeze(1)  

        curr_qs = self.action_net(states).gather(1, actions)  
        with torch.no_grad():
            next_actions = torch.argmax(self.action_net(next_state), dim=1, keepdim=True)
            next_qs = self.q_net(next_state).gather(1, next_actions)
            target_qs = rewards + self.gamma * next_qs * (1 - dones)

        loss = self.criterion(curr_qs, target_qs)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()  

    def hyperparam_update(self) -> None:
        self.tau=self.tau_update(self.tau, self.tau_decay)
        self.gamma=self.gamma_update(self.gamma, self.gamma_decay)
        self.epsilon=self.epsilon_update(self.epsilon, self.epsilon_min, self.epsilon_decay)
        self.LR=self.LR_update(self.LR, self.LR_decay)

    def train_step(self):
        for episode in range(self.episode_num):
            state, _ = self.env.reset()
            episode_reward = 0
            total_loss = 0
            steps_counted = 0

            for step in range(self.max_steps):
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward

                exp = Transition(state, action, reward, next_state, done)
                self.buffer.push(exp)

                self._step_counter += 1
                loss_val = self.update_net()
                if loss_val is not None:
                    total_loss += loss_val
                    steps_counted += 1

                if self._step_counter % self.update_step_interval == 0:
                    self.soft_update_target_net()

                state = next_state
                if done:
                    break

            self.episode_rewards.append(episode_reward)
            self.episode_losses.append(total_loss / steps_counted if steps_counted > 0 else float('nan'))
            self.episode_steps.append(step + 1)
            self.epsilon_history.append(self.epsilon)
            self.gamma_history.append(self.gamma)
            self.tau_history.append(self.tau)
            self.lr_history.append(self.LR)

            self.hyperparam_update()

            avg_loss = total_loss / steps_counted if steps_counted > 0 else float('nan')
            avg_reward = episode_reward / (step + 1)
            print(f"Episode {episode} => Average Step Loss: {avg_loss:.4f} | Episode Reward: {episode_reward} | Steps: {step + 1}")

        self.plot_dashboard(save_path=self.dash_path)

    def soft_update_target_net(self):
        action_net_state_dict = self.action_net.state_dict()
        q_net_state_dict = self.q_net.state_dict()
        for key in action_net_state_dict:
            q_net_state_dict[key] = action_net_state_dict[key]*self.tau + q_net_state_dict[key]*(1-self.tau)
        self.q_net.load_state_dict(action_net_state_dict)

    def plot_dashboard(self, save_path: str = None, show: bool = True, ma_window: int = 20):
        """
        Renders a single dashboard figure summarizing training progress, built
        entirely from the metric arrays accumulated during train_step():
        episode_rewards, episode_losses, episode_steps, and the per-episode
        hyperparameter histories (epsilon, gamma, tau, learning rate).
 
        Panels:
          1. Episode reward (raw + moving average)
          2. Training loss (raw + moving average)
          3. Reward distribution histogram
          4. Hyperparameter decay curves (epsilon / gamma / tau)
          5. Learning rate + episode length (steps to done/truncation)
 
        Args:
            save_path: optional path to save the figure (e.g. "dashboard.png").
            show: whether to call plt.show() at the end.
            ma_window: window size for moving-average smoothing.
 
        Returns:
            the matplotlib Figure object.
        """
 
        if len(self.episode_rewards) == 0:
            raise ValueError("No training metrics recorded yet — run train_step() first.")
 
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['figure.facecolor'] = 'white'
 
        episodes = np.arange(1, len(self.episode_rewards) + 1)
        rewards = np.array(self.episode_rewards, dtype=float)
        losses = np.array(self.episode_losses, dtype=float)
 
        def moving_average(x, window):
            window = max(1, min(window, len(x)))
            return np.convolve(x, np.ones(window) / window, mode='valid')
 
        colors = {
            'reward': '#3498db', 'reward_ma': '#e74c3c',
            'loss': '#9b59b6', 'loss_ma': '#e67e22',
            'epsilon': '#1abc9c', 'gamma': '#f39c12', 'tau': '#e91e63',
            'lr': '#2ecc71', 'steps': '#95a5a6',
        }
 
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)
        fig.suptitle("DDQN Training Dashboard", fontsize=20, fontweight='bold', color='#2c3e50', y=0.98)
 
        def style_axis(ax):
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(alpha=0.2)
 
        # 1. Reward per episode + moving average (full width)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(episodes, rewards, color=colors['reward'], alpha=0.35, linewidth=1, label='Episode Reward')
        if len(rewards) >= 2:
            ma = moving_average(rewards, ma_window)
            ax1.plot(episodes[-len(ma):], ma, color=colors['reward_ma'], linewidth=2.5, label=f'Moving Avg ({ma_window})')
        ax1.set_title("Episode Reward", fontsize=13, fontweight='bold', loc='left')
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Reward")
        ax1.legend(frameon=False)
        style_axis(ax1)
 
        # 2. Loss per episode
        ax2 = fig.add_subplot(gs[1, 0])
        mask = ~np.isnan(losses)
        if mask.any():
            ax2.plot(episodes[mask], losses[mask], color=colors['loss'], alpha=0.4, linewidth=1, label='Avg Step Loss')
            if mask.sum() >= 2:
                ma_loss = moving_average(losses[mask], ma_window)
                ax2.plot(episodes[mask][-len(ma_loss):], ma_loss, color=colors['loss_ma'], linewidth=2.5, label=f'Moving Avg ({ma_window})')
        ax2.set_title("Training Loss", fontsize=13, fontweight='bold', loc='left')
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Loss")
        ax2.legend(frameon=False)
        style_axis(ax2)
 
        # 3. Reward distribution
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.hist(rewards, bins=min(30, max(5, len(rewards) // 2)), color=colors['reward'], edgecolor='white', alpha=0.85)
        ax3.axvline(rewards.mean(), color=colors['reward_ma'], linestyle='--', linewidth=2, label=f"Mean: {rewards.mean():.1f}")
        ax3.set_title("Reward Distribution", fontsize=13, fontweight='bold', loc='left')
        ax3.set_xlabel("Reward")
        ax3.set_ylabel("Count")
        ax3.legend(frameon=False)
        style_axis(ax3)
 
        # 4. Hyperparameter decay
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(episodes, self.epsilon_history, color=colors['epsilon'], linewidth=2, label='Epsilon')
        ax4.plot(episodes, self.gamma_history, color=colors['gamma'], linewidth=2, label='Gamma')
        ax4.plot(episodes, self.tau_history, color=colors['tau'], linewidth=2, label='Tau')
        ax4.set_title("Hyperparameter Decay", fontsize=13, fontweight='bold', loc='left')
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Value")
        ax4.legend(frameon=False)
        style_axis(ax4)
 
        # 5. Learning rate + episode length
        ax5 = fig.add_subplot(gs[2, 1])
        ax5b = ax5.twinx()
        l1, = ax5.plot(episodes, self.lr_history, color=colors['lr'], linewidth=2, label='Learning Rate')
        l2, = ax5b.plot(episodes, self.episode_steps, color=colors['steps'], linewidth=1.5, alpha=0.7, label='Steps/Episode')
        ax5.set_title("Learning Rate & Episode Length", fontsize=13, fontweight='bold', loc='left')
        ax5.set_xlabel("Episode")
        ax5.set_ylabel("Learning Rate", color=colors['lr'])
        ax5b.set_ylabel("Steps", color=colors['steps'])
        ax5.tick_params(axis='y', labelcolor=colors['lr'])
        ax5b.tick_params(axis='y', labelcolor=colors['steps'])
        ax5b.spines['top'].set_visible(False)
        style_axis(ax5)
        ax5.legend(handles=[l1, l2], labels=['Learning Rate', 'Steps/Episode'], frameon=False, loc='upper right')
 
        last_n = min(20, len(rewards))
        fig.text(
            0.5, 0.01,
            f"Episodes: {len(episodes)}   |   Best Reward: {rewards.max():.1f}   |   "
            f"Avg Reward (last {last_n}): {rewards[-last_n:].mean():.1f}   |   "
            f"Final Epsilon: {self.epsilon_history[-1]:.3f}",
            ha='center', fontsize=11, color='#7f8c8d'
        )
 
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        if show:
            plt.show()
 
        return fig
