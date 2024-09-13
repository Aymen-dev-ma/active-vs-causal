# agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from cvae_scm import CVAE, SCM
import numpy as np
from torch import optim
from torch.nn import functional as F
from mcts import MCTSNode, MCTS
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

class CausalAgent(BaseAgent):
    def __init__(self, action_space, state_dim):
        super(CausalAgent, self).__init__(action_space, state_dim)
        self.action_dim = len(action_space)
        self.cvae = CVAE(input_dim=state_dim, latent_dim=20, condition_dim=self.action_dim)
        self.scm = SCM(state_dim=state_dim, action_dim=self.action_dim)
        self.cvae_optimizer = optim.Adam(self.cvae.parameters(), lr=1e-3)
    
    def select_action(self, state):
        # Use SCM to compute counterfactual outcomes for each action
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        expected_rewards = []
        for action in self.action_space:
            action_tensor = self.action_to_tensor(action).unsqueeze(0)
            counterfactual_state = self.scm.counterfactual(state_tensor, action_tensor)
            reward_estimate = self.scm.predict_reward(counterfactual_state)
            expected_rewards.append(reward_estimate)
    
        # Choose action with highest expected reward
        action = self.action_space[np.argmax(expected_rewards)]
        return action
    
    def update_model(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensors = torch.FloatTensor(states)
        actions_tensors = torch.stack([self.action_to_tensor(a) for a in actions])
        rewards_tensors = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_tensors = torch.FloatTensor(next_states)
    
        # Update CVAE model
        self.cvae_optimizer.zero_grad()
        recon_next_states, mu, logvar = self.cvae(states_tensors, actions_tensors)
        cvae_loss = self.cvae.loss_function(recon_next_states, next_states_tensors, mu, logvar)
        cvae_loss.backward()
        self.cvae_optimizer.step()
    
        # Update SCM based on new data
        self.scm.update(states_tensors, actions_tensors, next_states_tensors, rewards_tensors)
    
    def save_model(self, path):
        torch.save({
            'cvae_state_dict': self.cvae.state_dict(),
            'cvae_optimizer_state_dict': self.cvae_optimizer.state_dict(),
            'scm_state_dict': self.scm.state_dict(),
            'pyro_param_store': self.scm.get_pyro_param_store(),
        }, path)
    
    def action_to_tensor(self, action):
        # Convert action index to one-hot tensor
        action_tensor = torch.zeros(self.action_dim)
        action_tensor[action] = 1.0
        return action_tensor

class ActiveInferenceAgent(BaseAgent):
    def __init__(self, action_space, state_dim):
        super(ActiveInferenceAgent, self).__init__(action_space, state_dim)
        self.action_dim = len(action_space)
        self.generative_model = CVAE(input_dim=state_dim, latent_dim=20, condition_dim=self.action_dim)
        self.optimizer = optim.Adam(self.generative_model.parameters(), lr=1e-3)
        self.prior_preferences = torch.zeros(state_dim)
    
    def select_action(self, state):
        # Compute expected free energy for each action
        efe = []
        for action in self.action_space:
            expected_state = self.predict_state(state, action)
            efe_action = self.compute_expected_free_energy(expected_state)
            efe.append(efe_action)
    
        # Choose action that minimizes expected free energy
        action = self.action_space[np.argmin(efe)]
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
        kl_divergence = F.kl_div(predicted_state_tensor.log_softmax(dim=0), self.prior_preferences.softmax(dim=0), reduction='batchmean')
        return kl_divergence.item()
    
    def update_model(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert lists of numpy arrays to single numpy arrays
        states_np = np.array(states)
        actions_np = np.array(actions)
        next_states_np = np.array(next_states)
        rewards_np = np.array(rewards)
        dones_np = np.array(dones)
        
        # Convert numpy arrays to PyTorch tensors
        states_tensors = torch.FloatTensor(states_np)
        actions_tensors = torch.stack([self.action_to_tensor(a) for a in actions_np])
        next_states_tensors = torch.FloatTensor(next_states_np)
        rewards_tensors = torch.FloatTensor(rewards_np)
        dones_tensors = torch.FloatTensor(dones_np)
    
        # Update generative model
        self.optimizer.zero_grad()
        recon_next_states, mu, logvar = self.generative_model(states_tensors, actions_tensors)
        loss = self.generative_model.loss_function(recon_next_states, next_states_tensors, mu, logvar)
        loss.backward()
        self.optimizer.step()
    
    def save_model(self, path):
        torch.save({
            'generative_model_state_dict': self.generative_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def action_to_tensor(self, action):
        # Convert action index to one-hot tensor
        action_tensor = torch.zeros(self.action_dim)
        action_tensor[action] = 1.0
        return action_tensor
class CausalMCTSAgent(BaseAgent):
    def __init__(self, action_space, state_dim):
        super(CausalMCTSAgent, self).__init__(action_space, state_dim)
        self.action_dim = len(action_space)
        self.cvae = CVAE(input_dim=state_dim, latent_dim=20, condition_dim=self.action_dim)
        self.scm = SCM(state_dim=state_dim, action_dim=self.action_dim)
        self.mcts = MCTS(self.scm, self.action_space)
        self.cvae_optimizer = optim.Adam(self.cvae.parameters(), lr=1e-3)
    
    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.mcts.search(state_tensor)
        return action
    
    def update_model(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensors = torch.FloatTensor(states)
        actions_tensors = torch.stack([self.action_to_tensor(a) for a in actions])
        rewards_tensors = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_tensors = torch.FloatTensor(next_states)
    
        # Update CVAE model
        self.cvae_optimizer.zero_grad()
        recon_next_states, mu, logvar = self.cvae(states_tensors, actions_tensors)
        cvae_loss = self.cvae.loss_function(recon_next_states, next_states_tensors, mu, logvar)
        cvae_loss.backward()
        self.cvae_optimizer.step()
    
        # Update SCM based on new data
        self.scm.update(states_tensors, actions_tensors, next_states_tensors, rewards_tensors)
    
    def save_model(self, path):
        torch.save({
            'cvae_state_dict': self.cvae.state_dict(),
            'cvae_optimizer_state_dict': self.cvae_optimizer.state_dict(),
            'scm_state_dict': self.scm.state_dict(),
            'pyro_param_store': self.scm.get_pyro_param_store(),
        }, path)
    
    def action_to_tensor(self, action):
        # Convert action index to one-hot tensor
        action_tensor = torch.zeros(self.action_dim)
        action_tensor[action] = 1.0
        return action_tensor
    
    class CausalCuriosityAgent(BaseAgent):
        def __init__(self, action_space, state_dim):
            super(CausalCuriosityAgent, self).__init__(action_space, state_dim)
            self.action_dim = len(action_space)
            self.network = CausalCuriosityNetwork(state_dim=state_dim, action_dim=self.action_dim)
            self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3)
            self.gamma = 0.99  # Discount factor

        def select_action(self, state):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.network.policy(state_tensor)
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample().item()
            return action

        def update_model(self, batch):
            states, actions, rewards, next_states, dones = zip(*batch)
            states_tensors = torch.FloatTensor(states)
            actions_tensors = torch.LongTensor(actions)
            rewards_tensors = torch.FloatTensor(rewards)
            next_states_tensors = torch.FloatTensor(next_states)
            dones_tensors = torch.FloatTensor(dones)

            # Compute intrinsic rewards (curiosity)
            intrinsic_rewards = self.network.compute_intrinsic_rewards(states_tensors, actions_tensors, next_states_tensors)

            # Combine extrinsic and intrinsic rewards
            total_rewards = rewards_tensors + intrinsic_rewards

            # Update the network
            self.optimizer.zero_grad()
            loss = self.network.compute_loss(states_tensors, actions_tensors, total_rewards,
                                            next_states_tensors, dones_tensors, gamma=self.gamma)
            loss.backward()
            self.optimizer.step()

        def save_model(self, path):
            torch.save({
                'network_state_dict': self.network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, path)