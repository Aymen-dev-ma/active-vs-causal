# agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from cvae_scm import CVAE, SCM
import torch.nn.functional as F
import numpy as np
import time

class BaseAgent:
    def __init__(self, action_space, state_dim):
        self.action_space = action_space
        self.state_dim = state_dim

    def select_action(self, state):
        raise NotImplementedError

    def update_model(self, batch):
        raise NotImplementedError

    def save_model(self, path):
        raise NotImplementedError

    def action_to_tensor(self, action):
        # Convert action index to one-hot tensor
        action_tensor = torch.zeros(len(self.action_space))
        action_tensor[action] = 1.0
        return action_tensor

class CausalAgent(BaseAgent):
    def __init__(self, action_space, state_dim):
        super(CausalAgent, self).__init__(action_space, state_dim)
        self.action_dim = len(action_space)
        self.cvae = CVAE(input_dim=state_dim, latent_dim=20, condition_dim=self.action_dim)
        print(f"Initialized CVAE fc1 with input_dim={state_dim + self.action_dim}, output_dim=256")
        self.scm = SCM(state_dim=state_dim, action_dim=self.action_dim)
        self.cvae_optimizer = optim.Adam(self.cvae.parameters(), lr=1e-3)
        self.trained = False  # Flag to indicate if SCM has been trained
    
    def select_action(self, state):
        # If the SCM has not been trained yet, select a random action
        if not self.trained:
            return np.random.choice(self.action_space)
        # Use SCM to compute counterfactual outcomes for each action
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        expected_rewards = []
        for action in self.action_space:
            action_tensor = self.action_to_tensor(action).unsqueeze(0)
            counterfactual_state = self.scm.counterfactual(state_tensor, action_tensor)
            if counterfactual_state is None:
                # If counterfactual failed, select a random action
                return np.random.choice(self.action_space)
            reward_estimate = self.scm.predict_reward(counterfactual_state)
            expected_rewards.append(reward_estimate)
        # Choose action with highest expected reward
        action = self.action_space[np.argmax(expected_rewards)]
        return action
    
    def update_model(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensors = torch.FloatTensor(np.array(states))
        actions_tensors = torch.stack([self.action_to_tensor(a) for a in actions])
        rewards_tensors = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_tensors = torch.FloatTensor(np.array(next_states))
    
        # Update CVAE model
        self.cvae_optimizer.zero_grad()
        recon_next_states, mu, logvar = self.cvae(states_tensors, actions_tensors)
        cvae_loss = self.cvae.loss_function(recon_next_states, next_states_tensors, mu, logvar)
        cvae_loss.backward()
        self.cvae_optimizer.step()
    
        # Update SCM based on new data
        self.scm.update(states_tensors, actions_tensors, next_states_tensors, rewards_tensors)
        self.trained = True  # Set trained flag to True
    
    def save_model(self, path):
        torch.save({
            'cvae_state_dict': self.cvae.state_dict(),
            'cvae_optimizer_state_dict': self.cvae_optimizer.state_dict(),
            'scm_state_dict': self.scm.state_dict(),
            'pyro_param_store': self.scm.get_pyro_param_store(),
        }, path)

class ActiveInferenceAgent(BaseAgent):
    def __init__(self, action_space, state_dim):
        super(ActiveInferenceAgent, self).__init__(action_space, state_dim)
        self.action_dim = len(action_space)
        print(f"Initializing ActiveInferenceAgent with state_dim={state_dim}, action_dim={self.action_dim}")
        self.generative_model = CVAE(input_dim=state_dim, latent_dim=20, condition_dim=self.action_dim)
        self.optimizer = optim.Adam(self.generative_model.parameters(), lr=1e-3)
        self.prior_preferences = torch.zeros(state_dim)
    
    def select_action(self, state):
        start_time = time.time()
        # Compute expected free energy for each action
        efe = []
        for action in self.action_space:
            expected_state = self.predict_state(state, action)
            efe_action = self.compute_expected_free_energy(expected_state)
            efe.append(efe_action)
        # Choose action that minimizes expected free energy
        action = self.action_space[np.argmin(efe)]
        end_time = time.time()
        print(f"select_action took {end_time - start_time:.4f} seconds")
        return action
    
    def predict_state(self, state, action):
        # Use generative model to predict next state
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = self.action_to_tensor(action).unsqueeze(0)
        predicted_state, _, _ = self.generative_model(state_tensor, action_tensor)
        return predicted_state.detach().squeeze(0).numpy()
    
    def compute_expected_free_energy(self, predicted_state):
        # Compute the divergence between predicted state and prior preferences
        predicted_state_tensor = torch.FloatTensor(predicted_state)
        # Using mean squared error as a proxy for KL divergence
        efe = F.mse_loss(predicted_state_tensor, self.prior_preferences, reduction='sum')
        return efe.item()
    
    def update_model(self, batch):
        start_time = time.time()
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensors = torch.FloatTensor(np.array(states))
        actions_tensors = torch.stack([self.action_to_tensor(a) for a in actions])
        next_states_tensors = torch.FloatTensor(np.array(next_states))
    
        # Update generative model
        self.optimizer.zero_grad()
        recon_next_states, mu, logvar = self.generative_model(states_tensors, actions_tensors)
        loss = self.generative_model.loss_function(recon_next_states, next_states_tensors, mu, logvar)
        loss.backward()
        self.optimizer.step()
        end_time = time.time()
        print(f"update_model took {end_time - start_time:.4f} seconds")
    
    def save_model(self, path):
        torch.save({
            'generative_model_state_dict': self.generative_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
