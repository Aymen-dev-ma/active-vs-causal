# environment.py

import time
import numpy as np
import torch
import os

class GameEnvironment:
    def __init__(self, games_no=1, max_steps_per_episode=100):
        self.games_no = games_no
        self.max_steps_per_episode = max_steps_per_episode
        self.current_step = 0  # Initialize step counter

        # Load dataset
        current_time = time.time()
        dataset_path = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset not found at {dataset_path}")

        dataset = np.load(dataset_path, allow_pickle=True, encoding='latin1')
        self.imgs = torch.from_numpy(dataset['imgs']).float()  # Shape: (737280, 64, 64)
        self.latents_values = torch.from_numpy(dataset['latents_values']).float()
        self.latents_classes = torch.from_numpy(dataset['latents_classes']).long()
        metadata = dataset['metadata'][()]
        self.s_sizes = torch.tensor(metadata['latents_sizes'])  # [1, 3, 6, 40, 32, 32]
        self.s_dim = self.s_sizes.size(0) + 1  # +1 for reward

        # Initialize current_s with the correct dimensions
        self.current_s = torch.zeros((self.games_no, self.s_dim), dtype=torch.float32)
        self.last_r = torch.zeros(self.games_no, dtype=torch.float32)
        self.done = [False] * self.games_no  # Initialize the done attribute
        self.new_image_all()

        # Precompute s_bases for indexing
        self.s_bases = torch.cat((torch.tensor([1]), self.s_sizes[:-1])).cumprod(0)

        print('Dataset loaded. Time:', time.time() - current_time, 'datapoints:', len(self.imgs), 's_dim:', self.s_dim)

    def reset(self):
        self.randomize_environment_all()
        self.current_step = 0  # Reset step counter
        state = self.current_s[0].numpy()
        return state

    def step(self, action, index=0):
        # Update the state based on the action
        if action == 0:  # Up
            self.current_s[index, 0] += 1
        elif action == 1:  # Down
            self.current_s[index, 0] -= 1
        elif action == 2:  # Left
            self.current_s[index, 1] -= 1
        elif action == 3:  # Right
            self.current_s[index, 1] += 1

        # Ensure state values are within valid ranges
        self.current_s[index, :self.s_dim - 1] = torch.clamp(self.current_s[index, :self.s_dim - 1], min=0)

        # Calculate the reward based on the new state
        reward = -torch.sum(torch.abs(self.current_s[index])).item()

        # Increment step counter
        self.current_step += 1

        # Check if the game is done
        done = False
        if self.current_step >= self.max_steps_per_episode:
            done = True

        # Update the done status
        self.done[index] = done

        # Return the new state, reward, and done flag
        return self.current_s[index].numpy(), reward, done

    def randomize_environment_all(self):
        # Randomize the environment
        for i in range(self.games_no):
            self.current_s[i, :self.s_dim - 1] = torch.randint(0, 10, (self.s_dim - 1,), dtype=torch.float32)

    def new_image_all(self):
        pass  # Placeholder for any image-related functionality
