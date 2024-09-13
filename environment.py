# environment.py

import time
import numpy as np
import torch
import os

class GameEnvironment:
    def __init__(self, games_no=1):
        self.games_no = games_no

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
        self.new_image_all()

        # Precompute s_bases for indexing
        self.s_bases = torch.cat((torch.tensor([1]), self.s_sizes[:-1])).cumprod(0)

        print('Dataset loaded. Time:', time.time() - current_time, 'datapoints:', len(self.imgs), 's_dim:', self.s_dim)

    def reset(self):
        self.randomize_environment_all()
        state = self.current_frame_all()
        state = state.view(-1).numpy()
        return state

    def step(self, action, index=0):
        # Aktualisiere den Zustand basierend auf der Aktion
        if action == 0:  # Beispielaktion: nach oben
            self.current_s[index, 0] += 1
        elif action == 1:  # Beispielaktion: nach unten
            self.current_s[index, 0] -= 1
        elif action == 2:  # Beispielaktion: nach links
            self.current_s[index, 1] -= 1
        elif action == 3:  # Beispielaktion: nach rechts
            self.current_s[index, 1] += 1

        # Berechne die Belohnung basierend auf dem neuen Zustand
        reward = -np.sum(np.abs(self.current_s[index]))

        # Überprüfe, ob das Spiel beendet ist
        done = np.all(self.current_s[index] == 0)

        # Aktualisiere den Abschlussstatus
        self.done[index] = done

        return self.current_s[index], reward, done

    def sample_s(self):  # Reward is zero after this!
        s = torch.zeros(self.s_dim - 1, dtype=torch.float32)
        for s_i, s_size in enumerate(self.s_sizes):
            s[s_i] = torch.randint(0, s_size.item(), (1,))
        return s

    def sample_s_all(self):  # Reward is zero after this!
        s = torch.zeros((self.games_no, self.s_dim - 1), dtype=torch.float32)
        for s_i in range(self.s_sizes.size(0)):
            s[:, s_i] = torch.randint(0, self.s_sizes[s_i].item(), (self.games_no,), dtype=torch.float32)
        return s

    def s_to_index(self, s):
        indices = (s.long() * self.s_bases).sum(dim=1)
        return indices

    def s_to_o(self, index):
        indices = self.s_to_index(self.current_s[:, :-1])
        images = self.imgs[indices]
        return images

    def current_frame(self, index):
        return self.s_to_o(index)

    def current_frame_all(self):
        return self.s_to_o(None)  # index is not used

    def randomize_environment(self, index):
        self.current_s[index, :-1] = self.sample_s()
        self.current_s[index, -1] = -10 + torch.rand(1).item() * 20
        self.last_r[index] = -1.0 + torch.rand(1).item() * 2.0

    def randomize_environment_all(self):
        self.current_s[:, :-1] = self.sample_s_all()
        self.current_s[:, -1] = -10 + torch.rand(self.games_no) * 20
        self.last_r = -1.0 + torch.rand(self.games_no) * 2.0

    def new_image(self, index):
        reward = self.current_s[index, -1]  # pass reward to the new latent..!
        self.current_s[index, :-1] = self.sample_s()
        self.current_s[index, -1] = reward

    def new_image_all(self):
        reward = self.current_s[:, -1] if self.current_s.shape[1] > 6 else torch.zeros(self.games_no)
        self.current_s[:, :-1] = self.sample_s_all()
        if self.current_s.shape[1] > 6:
            self.current_s[:, -1] = reward
        # print(f"Shape of self.current_s: {self.current_s.shape}")  # Debugging line

    def get_reward(self, index):
        return self.current_s[index, -1]

    def pi_to_action(self, pi, index, repeats=1):
        for _ in range(repeats):
            if pi == 0:
                self.up(index)
            elif pi == 1:
                self.down(index)
            elif pi == 2:
                self.left(index)
            elif pi == 3:
                self.right(index)
            else:
                raise ValueError('Invalid action')

    def tick(self, index):
        self.last_r[index] *= 0.95

    def tick_all(self):
        self.last_r *= 0.95

    def up(self, index):
        self.tick(index)
        self.current_s[index, 5] += 1.0
        if self.current_s[index, 5] >= 32:
            self.new_image(index)

    def down(self, index):
        self.tick(index)
        if self.current_s[index, 5] > 0:
            self.current_s[index, 5] -= 1.0

    def left(self, index):
        self.tick(index)
        if self.current_s[index, 4] < 31:
            self.current_s[index, 4] += 1.0

    def right(self, index):
        self.tick(index)
        if self.current_s[index, 4] > 0:
            self.current_s[index, 4] -= 1.0
