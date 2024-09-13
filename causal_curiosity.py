# causal_curiosity.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalCuriosityNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CausalCuriosityNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Policy network
        self.policy_fc1 = nn.Linear(state_dim, 128)
        self.policy_fc2 = nn.Linear(128, action_dim)

        # Value network for baseline
        self.value_fc1 = nn.Linear(state_dim, 128)
        self.value_fc2 = nn.Linear(128, 1)

        # Forward dynamics model (for curiosity)
        self.forward_fc1 = nn.Linear(state_dim + action_dim, 128)
        self.forward_fc2 = nn.Linear(128, state_dim)

        # Inverse dynamics model (for deconfounding)
        self.inverse_fc1 = nn.Linear(state_dim * 2, 128)
        self.inverse_fc2 = nn.Linear(128, action_dim)

    def policy(self, state):
        x = F.relu(self.policy_fc1(state))
        action_logits = self.policy_fc2(x)
        action_probs = F.softmax(action_logits, dim=1)
        return action_probs

    def value(self, state):
        x = F.relu(self.value_fc1(state))
        state_value = self.value_fc2(x)
        return state_value.squeeze(1)

    def forward_dynamics(self, state, action):
        x = torch.cat([state, self.one_hot_action(action)], dim=1)
        x = F.relu(self.forward_fc1(x))
        next_state_pred = self.forward_fc2(x)
        return next_state_pred

    def inverse_dynamics(self, state, next_state):
        x = torch.cat([state, next_state], dim=1)
        x = F.relu(self.inverse_fc1(x))
        action_logits = self.inverse_fc2(x)
        return action_logits

    def compute_intrinsic_rewards(self, states, actions, next_states):
        # Predict next state using forward dynamics model
        next_state_pred = self.forward_dynamics(states, actions)
        # Compute curiosity as the error between predicted and actual next state
        intrinsic_rewards = 0.5 * (next_states - next_state_pred).pow(2).mean(dim=1)
        return intrinsic_rewards.detach()

    def compute_loss(self, states, actions, rewards, next_states, dones, gamma):
        # Compute state values and advantages
        state_values = self.value(states)
        next_state_values = self.value(next_states)
        returns = rewards + gamma * next_state_values * (1 - dones)
        advantages = returns - state_values.detach()

        # Policy loss
        action_probs = self.policy(states)
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + 1e-8)
        policy_loss = -torch.mean(action_log_probs * advantages)

        # Value loss
        value_loss = F.mse_loss(state_values, returns.detach())

        # Forward dynamics loss (for curiosity)
        next_state_pred = self.forward_dynamics(states, actions)
        forward_loss = F.mse_loss(next_state_pred, next_states.detach())

        # Inverse dynamics loss (for deconfounding)
        action_logits = self.inverse_dynamics(states, next_states)
        inverse_loss = F.cross_entropy(action_logits, actions)

        # Total loss
        total_loss = policy_loss + 0.5 * value_loss + 0.1 * forward_loss + 0.1 * inverse_loss

        return total_loss

    def one_hot_action(self, action):
        # Convert action indices to one-hot vectors
        action_one_hot = torch.zeros(action.size(0), self.action_dim)
        action_one_hot[range(action.size(0)), action] = 1.0
        return action_one_hot
