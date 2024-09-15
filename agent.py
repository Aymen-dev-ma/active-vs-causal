# agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from cvae_scm import CVAE, SCM
import torch.nn.functional as F
import numpy as np
import time
from frontdoor_scm import FrontdoorCVAE, FrontdoorSCM
#from mcts import CausalUCTNode
import math
from cvae_scm import CVAE, SCM, CEVAE
import random
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal  # Import AutoGuide

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
class CEVAEAgent(BaseAgent):
    def __init__(self, action_space, state_dim):
        super(CEVAEAgent, self).__init__(action_space, state_dim)
        self.action_dim = len(action_space)
        self.cevae = CEVAE(input_dim=state_dim, latent_dim=20, treatment_dim=self.action_dim)
        self.optimizer = optim.Adam(self.cevae.parameters(), lr=1e-3)
        self.trained = False  # Flag to indicate if CEVAE has been trained

    def select_action(self, state):
        # If the CEVAE has not been trained yet, select a random action
        if not self.trained:
            return np.random.choice(self.action_space)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        expected_rewards = []
        for action in self.action_space:
            action_tensor = self.action_to_tensor(action).unsqueeze(0)
            # Estimate the expected outcome for each action using CEVAE
            reward_estimate = self.cevae.predict_counterfactual(state_tensor, action_tensor)
            expected_rewards.append(reward_estimate.item())
        # Choose action with highest expected reward
        action = self.action_space[np.argmax(expected_rewards)]
        return action

    def update_model(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensors = torch.FloatTensor(np.array(states))
        actions_tensors = torch.stack([self.action_to_tensor(a) for a in actions])
        rewards_tensors = torch.FloatTensor(rewards).unsqueeze(1)
        
        # Train the CEVAE model
        self.optimizer.zero_grad()
        loss = self.cevae.loss_function(states_tensors, actions_tensors, rewards_tensors)
        loss.backward()
        self.optimizer.step()
        self.trained = True

    def save_model(self, path):
        torch.save({
            'cevae_state_dict': self.cevae.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
class IPSAgent(BaseAgent):
    def __init__(self, action_space, state_dim):
        super(IPSAgent, self).__init__(action_space, state_dim)
        self.action_dim = len(action_space)
        # Propensity model: predicts probability of each action given state
        self.propensity_model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim),
            nn.Softmax(dim=1)
        )
        # Reward model: predicts expected reward given state and action
        self.reward_model = nn.Sequential(
            nn.Linear(state_dim + self.action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.optimizer_propensity = optim.Adam(self.propensity_model.parameters(), lr=1e-3)
        self.optimizer_reward = optim.Adam(self.reward_model.parameters(), lr=1e-3)
        self.trained = False

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        if not self.trained:
            return np.random.choice(self.action_space)
        else:
            with torch.no_grad():
                # Estimate expected reward for each action
                expected_rewards = []
                propensity_scores = self.propensity_model(state_tensor)  # Shape: [1, action_dim]
                for action in self.action_space:
                    action_tensor = self.action_to_tensor(action).unsqueeze(0)
                    sa = torch.cat([state_tensor, action_tensor], dim=1)
                    reward_pred = self.reward_model(sa)  # Shape: [1, 1]
                    propensity = propensity_scores[0, action]
                    if propensity.item() < 1e-6:
                        propensity = torch.tensor(1e-6)
                    ips_reward = reward_pred.item()  # For action selection, use predicted reward
                    expected_rewards.append(ips_reward)
                # Select action with highest expected reward
                best_action = self.action_space[np.argmax(expected_rewards)]
                return best_action

    def update_model(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensors = torch.FloatTensor(np.array(states))
        actions_tensors = torch.LongTensor(actions)
        actions_one_hot = F.one_hot(actions_tensors, num_classes=self.action_dim).float()
        rewards_tensors = torch.FloatTensor(rewards).unsqueeze(1)

        # Update propensity model
        self.optimizer_propensity.zero_grad()
        logits = self.propensity_model[0:-1](states_tensors)
        propensity_loss = F.cross_entropy(logits, actions_tensors)
        propensity_loss.backward()
        self.optimizer_propensity.step()

        # Update reward model using IPS weighting
        self.optimizer_reward.zero_grad()
        sa = torch.cat([states_tensors, actions_one_hot], dim=1)
        reward_preds = self.reward_model(sa)
        with torch.no_grad():
            propensity_scores = self.propensity_model(states_tensors)
            action_probs = propensity_scores.gather(1, actions_tensors.unsqueeze(1))
            weights = 1.0 / (action_probs + 1e-6)
        weighted_loss = torch.mean(weights * F.mse_loss(reward_preds, rewards_tensors, reduction='none'))
        weighted_loss.backward()
        self.optimizer_reward.step()
        self.trained = True

    def save_model(self, path):
        torch.save({
            'propensity_model_state_dict': self.propensity_model.state_dict(),
            'reward_model_state_dict': self.reward_model.state_dict(),
            'optimizer_propensity_state_dict': self.optimizer_propensity.state_dict(),
            'optimizer_reward_state_dict': self.optimizer_reward.state_dict(),
        }, path)

class CausalUCTAgent(BaseAgent):
    def __init__(self, action_space, state_dim, uct_simulations=10):
        super(CausalUCTAgent, self).__init__(action_space, state_dim)
        self.action_dim = len(action_space)
        self.scm = SCM(state_dim=state_dim, action_dim=self.action_dim)

        # Initialize Pyro's optimizer and SVI
        self.pyro_optimizer = pyro.optim.Adam({"lr": 1e-3})
        self.svi = SVI(self.scm.model, self.scm.guide, self.pyro_optimizer, loss=Trace_ELBO())
        self.trained = False
        self.uct_simulations = uct_simulations
        self.c_puct = 1.0

        # Value function approximator
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=1e-3)

    def select_action(self, state):
        if not self.trained:
            return np.random.choice(self.action_space)
        root = UCTNode(state, None, None, self)
        for _ in range(self.uct_simulations):
            node = root
            state_copy = torch.FloatTensor(state).unsqueeze(0)
            path = []

            # Selection
            while not node.is_terminal():
                if not node.is_fully_expanded():
                    # Expansion
                    untried_actions = [a for a in self.action_space if a not in node.children]
                    action = random.choice(untried_actions)
                    action_tensor = self.action_to_tensor(action).unsqueeze(0)
                    next_state = self.scm.counterfactual(state_copy, action_tensor)
                    if next_state is None:
                        break
                    next_state_np = next_state.squeeze(0).numpy()
                    child_node = node.expand(action, next_state_np)
                    # Evaluate using value function
                    value = self.value_network(torch.FloatTensor(next_state_np)).item()
                    # Backpropagation
                    child_node.backpropagate(value)
                    break  # Only expand one node at a time
                else:
                    # Select best child
                    action, node = node.select_child(self.c_puct)
                    action_tensor = self.action_to_tensor(action).unsqueeze(0)
                    next_state = self.scm.counterfactual(state_copy, action_tensor)
                    if next_state is None:
                        break
                    state_copy = next_state
                    path.append((node, action))

            # Backpropagation
            # If node is terminal or leaf, use value function to estimate value
            if node.is_terminal():
                value = 0
            else:
                state_tensor = torch.FloatTensor(node.state).unsqueeze(0)
                value = self.value_network(state_tensor).item()
            node.backpropagate(value)

        best_action = root.best_action()
        return best_action

    def rollout(self, state):
        # No longer needed since we use the value function
        pass

    def update_model(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensors = torch.FloatTensor(np.array(states))
        actions_tensors = torch.stack([self.action_to_tensor(a) for a in actions])
        rewards_tensors = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_tensors = torch.FloatTensor(np.array(next_states))

        # Update SCM using SVI
        num_iterations = 1
        for _ in range(num_iterations):
            loss = self.svi.step(states_tensors, actions_tensors, next_states_tensors, rewards_tensors)
        
        # Update value network
        self.value_optimizer.zero_grad()
        # Compute target values using rewards and bootstrap from value network
        with torch.no_grad():
            next_values = self.value_network(next_states_tensors)
            targets = rewards_tensors + 0.99 * next_values * (1 - torch.FloatTensor(dones).unsqueeze(1))
        values = self.value_network(states_tensors)
        value_loss = F.mse_loss(values, targets)
        value_loss.backward()
        self.value_optimizer.step()

        self.trained = True

    def save_model(self, path):
        # Save Pyro's parameter store and value network state
        torch.save({
            'value_network_state_dict': self.value_network.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'pyro_param_store': pyro.get_param_store().get_state(),
        }, path)

class UCTNode:
    def __init__(self, state, parent, action, agent):
        self.state = state  # Numpy array
        self.parent = parent
        self.action = action
        self.agent = agent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.is_terminal_node = False

    def is_fully_expanded(self):
        return len(self.children) == len(self.agent.action_space)

    def is_terminal(self):
        # Define terminal condition if necessary
        return False  # Modify as per your environment

    def expand(self, action, next_state):
        child_node = UCTNode(next_state, self, action, self.agent)
        self.children[action] = child_node
        return child_node

    def select_child(self, c_puct):
        best_score = -float('inf')
        best_action = None
        best_child = None
        for action, child in self.children.items():
            q_value = child.total_value / (child.visit_count + 1e-5)
            u_value = c_puct * math.sqrt(math.log(self.visit_count + 1) / (child.visit_count + 1e-5))
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def backpropagate(self, value):
        self.visit_count += 1
        self.total_value += value
        if self.parent:
            self.parent.backpropagate(value)

    def best_action(self):
        # Return the action with the highest visit count
        best_visit = -float('inf')
        best_action = None
        for action, child in self.children.items():
            if child.visit_count > best_visit:
                best_visit = child.visit_count
                best_action = action
        return best_action
    
    
class FrontdoorAgent(BaseAgent):
    def __init__(self, action_space, state_dim, mcts_simulations=10):
        super(FrontdoorAgent, self).__init__(action_space, state_dim)
        self.action_dim = len(action_space)

        # Initialize FrontdoorCVAE
        self.cvae = FrontdoorCVAE(input_dim=state_dim, latent_dim=10, action_dim=self.action_dim)
        self.cvae_optimizer = optim.Adam(self.cvae.parameters(), lr=1e-3)

        # Initialize FrontdoorSCM
  # Initialize SCM
        self.scm = FrontdoorSCM(state_dim=state_dim, action_dim=self.action_dim, latent_dim=10)
        # Pyro optimizer and SVI for SCM
        self.pyro_optimizer = pyro.optim.Adam({"lr": 1e-3})
        # Use AutoNormal guide
        self.guide = AutoNormal(self.scm.model)
        self.svi = SVI(self.scm.model, self.guide, self.pyro_optimizer, loss=Trace_ELBO())

        self.trained = False
        self.mcts_simulations = mcts_simulations
        self.c_puct = 1.0

        # Value function approximator
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=1e-3)

    def select_action(self, state):
        if not self.trained:
            return np.random.choice(self.action_space)
        root = FrontdoorMCTSNode(state, None, None, self)
        for _ in range(self.mcts_simulations):
            node = root
            state_copy = torch.FloatTensor(state).unsqueeze(0)
            path = []

            # Selection and Expansion
            while not node.is_terminal():
                if not node.is_fully_expanded():
                    # Expansion
                    untried_actions = [a for a in self.action_space if a not in node.children]
                    action = random.choice(untried_actions)
                    action_tensor = self.action_to_tensor(action).unsqueeze(0)
                    # Generate next state using SCM and CVAE with front-door adjustment
                    next_state = self.generate_next_state(state_copy, action_tensor)
                    if next_state is None:
                        break
                    next_state_np = next_state.squeeze(0).numpy()
                    child_node = node.expand(action, next_state_np)
                    # Evaluate using value function
                    value = self.value_network(torch.FloatTensor(next_state_np)).item()
                    # Backpropagation
                    child_node.backpropagate(value)
                    break  # Only expand one node at a time
                else:
                    # Select best child
                    action, node = node.select_child(self.c_puct)
                    action_tensor = self.action_to_tensor(action).unsqueeze(0)
                    next_state = self.generate_next_state(state_copy, action_tensor)
                    if next_state is None:
                        break
                    state_copy = next_state
                    path.append((node, action))

            # Backpropagation
            if node.is_terminal():
                value = 0
            else:
                state_tensor = torch.FloatTensor(node.state).unsqueeze(0)
                value = self.value_network(state_tensor).item()
            node.backpropagate(value)

        best_action = root.best_action()
        return best_action

    def generate_next_state(self, s, a):
        # Use front-door adjustment to generate the next state
        with torch.no_grad():
            # Sample latent variable z using the CVAE encoder
            mu, logvar = self.cvae.encode(s)
            z = self.cvae.reparameterize(mu, logvar)
            # Generate next state using the SCM's counterfactual method
            s_next = self.scm.counterfactual(s, a, z)
            return s_next

    def update_model(self, batch):
        states, actions, rewards, next_states, dones = zip(*batch)
        states_tensors = torch.FloatTensor(np.array(states))
        actions_tensors = torch.stack([self.action_to_tensor(a) for a in actions])
        rewards_tensors = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_tensors = torch.FloatTensor(np.array(next_states))

        # Update CVAE
        self.cvae_optimizer.zero_grad()
        recon_batch, mu, logvar = self.cvae(states_tensors, actions_tensors)
        cvae_loss = self.cvae.loss_function(recon_batch, states_tensors, mu, logvar)
        cvae_loss.backward()
        self.cvae_optimizer.step()

        # Update SCM using SVI
        num_iterations = 1
        for _ in range(num_iterations):
            loss = self.svi.step(states_tensors, actions_tensors, next_states_tensors, rewards_tensors)

        # Update value network
        self.value_optimizer.zero_grad()
        with torch.no_grad():
            next_values = self.value_network(next_states_tensors)
            dones_tensors = torch.FloatTensor(dones).unsqueeze(1)
            targets = rewards_tensors + 0.99 * next_values * (1 - dones_tensors)
        values = self.value_network(states_tensors)
        value_loss = F.mse_loss(values, targets)
        value_loss.backward()
        self.value_optimizer.step()

        self.trained = True

    def save_model(self, path):
        torch.save({
            'cvae_state_dict': self.cvae.state_dict(),
            'cvae_optimizer_state_dict': self.cvae_optimizer.state_dict(),
            'value_network_state_dict': self.value_network.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
            'pyro_param_store': pyro.get_param_store().get_state(),
        }, path)

class FrontdoorMCTSNode:
    def __init__(self, state, parent, action, agent):
        self.state = state  # Numpy array
        self.parent = parent
        self.action = action
        self.agent = agent
        self.children = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.is_terminal_node = False

    def is_fully_expanded(self):
        return len(self.children) == len(self.agent.action_space)

    def is_terminal(self):
        # Define terminal condition if necessary
        return False

    def expand(self, action, next_state):
        child_node = FrontdoorMCTSNode(next_state, self, action, self.agent)
        self.children[action] = child_node
        return child_node

    def select_child(self, c_puct):
        best_score = -float('inf')
        best_action = None
        best_child = None
        for action, child in self.children.items():
            q_value = child.total_value / (child.visit_count + 1e-5)
            u_value = c_puct * math.sqrt(math.log(self.visit_count + 1) / (child.visit_count + 1e-5))
            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def backpropagate(self, value):
        self.visit_count += 1
        self.total_value += value
        if self.parent:
            self.parent.backpropagate(value)

    def best_action(self):
        # Return the action with the highest visit count
        best_visit = -float('inf')
        best_action = None
        for action, child in self.children.items():
            if child.visit_count > best_visit:
                best_visit = child.visit_count
                best_action = action
        return best_action
