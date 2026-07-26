from models.buffer import BaseBuffer
from models.experience import make_transition_class

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from collections.abc import Callable


Transition = make_transition_class()  # build the namedtuple class once


class DQN_trainer:
    def __init__(self, batch_size:int, episode_num:int, max_steps:int, update_step_interval:int,
                 tau:float, tau_decay:float, tau_update:Callable,
                 gamma:float, gamma_decay:float, gamma_update:Callable,
                 learning_rate:float, learning_rate_decay:float, learning_rate_update:Callable,
                 buffer:BaseBuffer, buffer_feature:str, action_dim:int,
                 epsilon:float, epsilon_min:float, epsilon_decay:float, epsilon_update:Callable,
                 policy_net:nn.Module, target_net:nn.Module, optim:optim.Optimizer, criterion:nn.modules.loss._Loss,
                 device:str, env) -> None:

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

        self.policy_net=policy_net
        self.target_net=target_net
        self.optim=optim
        self.criterion=criterion

        self.device=device
        self.env=env
        # --------------------------------
        self.episode_losses=[]
        self.episode_rewards=[]
        self._step_counter = 0  
        # --------------------------------

    def select_action(self, state) -> int:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        rand_val = torch.rand(1).item()
        with torch.no_grad():
            if rand_val <= self.epsilon:
                action = torch.randint(0, self.action_dim, (1,)).item()  
            else:
                action = torch.argmax(self.policy_net(state_t), dim=1).item()  

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

        curr_qs = self.policy_net(states).gather(1, actions)  
        with torch.no_grad():
            next_max_qs = torch.max(self.target_net(next_state), dim=1).values.unsqueeze(1)  
            target_qs = rewards + self.gamma * next_max_qs * (1 - dones)

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

            self.hyperparam_update()

            avg_loss = total_loss / steps_counted if steps_counted > 0 else float('nan')
            avg_reward = episode_reward / (step + 1)
            print(f"Episode {episode} => Average Step Loss: {avg_loss:.4f} | Episode Reward: {episode_reward} | Steps: {step + 1}")

    def soft_update_target_net(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)