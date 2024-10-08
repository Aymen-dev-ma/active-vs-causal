# mcts.py

import torch
import torch.nn.functional as F
import math

class MCTSNode:
    def __init__(self, state, parent, action, cvae, scm, action_dim):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.cvae = cvae
        self.scm = scm
        self.action_dim = action_dim

    def is_leaf(self):
        return len(self.children) == 0

    def expand(self):
        # Expand node by creating child nodes for each possible action
        for action in range(self.action_dim):
            next_state = self.simulate_action(self.state, action)
            self.children[action] = MCTSNode(
                state=next_state,
                parent=self,
                action=action,
                cvae=self.cvae,
                scm=self.scm,
                action_dim=self.action_dim
            )

    def simulate_action(self, state, action):
        # Simulate state transition using the CVAE and SCM
        state_tensor = state.to(self.cvae.device)
        action_tensor = F.one_hot(torch.tensor([action]), num_classes=self.action_dim).float().to(self.cvae.device)
        z, _, _ = self.cvae.encode(state_tensor, action_tensor)
        next_state = self.cvae.decode(z, action_tensor)
        return next_state

    def backpropagate(self, reward):
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def best_child(self, c_param=1.4):
        choices_weights = []
        for child in self.children.values():
            if child.visits == 0:
                ucb1 = float('inf')
            else:
                exploit = child.value / child.visits
                explore = c_param * math.sqrt(math.log(self.visits) / child.visits)
                ucb1 = exploit + explore
            choices_weights.append(ucb1)
        best_choice = torch.argmax(torch.tensor(choices_weights))
        return list(self.children.values())[best_choice]
    def evaluate_state(self, state):
        # Use the SCM to evaluate the state causally
        # For example, estimate the expected reward under different actions
        with torch.no_grad():
            state = state.to(self.cvae.device)
            current_action = F.one_hot(torch.tensor([0]), num_classes=self.action_dim).float().to(self.cvae.device)

            expected_rewards = []
            for action in range(self.action_dim):
                action_tensor = F.one_hot(torch.tensor([action]), num_classes=self.action_dim).float().to(self.cvae.device)
                effect = self.scm.causal_effect(state, current_action, current_action, action_tensor)
                expected_rewards.append(effect)

            # Return the maximum expected reward as the value of the state
            return max(expected_rewards)

class MCTS:
    def __init__(self, cvae, scm, action_dim):
        self.cvae = cvae
        self.scm = scm
        self.action_dim = action_dim

    def search(self, state, num_simulations=50):
        root = MCTSNode(
            state=state,
            parent=None,
            action=None,
            cvae=self.cvae,
            scm=self.scm,
            action_dim=self.action_dim
        )

        for _ in range(num_simulations):
            node = root
            # Selection
            while not node.is_leaf():
                node = node.best_child()
            # Expansion
            if node.visits > 0:
                node.expand()
                node = node.best_child()
            # Simulation
            reward = self.rollout(node.state)
            # Backpropagation
            node.backpropagate(reward)

        # Choose the action with the highest visit count
        best_action = max(root.children.items(), key=lambda item: item[1].visits)[0]
        return best_action

    def rollout(self, state, max_depth=5):
        # Simulate a random playout from the current state
        current_state = state
        total_reward = 0.0
        for _ in range(max_depth):
            action = torch.randint(0, self.action_dim, (1,)).item()
            next_state = self.simulate_action(current_state, action)
            reward = self.get_reward(next_state)
            total_reward += reward
            current_state = next_state
        return total_reward

    def simulate_action(self, state, action):
        # Get the device from the model's parameters
        device = next(self.cvae.parameters()).device

        state_tensor = state.to(device)
        action_tensor = F.one_hot(torch.tensor([action]), num_classes=self.action_dim).float().to(device)
        z, _, _ = self.cvae.encode(state_tensor, action_tensor)
        next_state = self.cvae.decode(z, action_tensor)
        return next_state

    def get_reward(self, state):
        # Simple reward function based on state (e.g., sum of pixel values)
        reward = -torch.abs(state.sum() - (self.cvae.image_dim / 2))
        return reward.item()
