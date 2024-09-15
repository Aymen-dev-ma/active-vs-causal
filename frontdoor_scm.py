# frontdoor_scm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer.autoguide import AutoNormal

class FrontdoorCVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, action_dim):
        super(FrontdoorCVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc21 = nn.Linear(128, latent_dim)  # Mean
        self.fc22 = nn.Linear(128, latent_dim)  # Log variance

        # Decoder
        self.fc3 = nn.Linear(latent_dim + action_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, a):
        concatenated = torch.cat([z, a], dim=1)
        h3 = F.relu(self.fc3(concatenated))
        return self.fc4(h3)

    def forward(self, x, a):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, a)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + KLD

class FrontdoorSCM(PyroModule):
    def __init__(self, state_dim, action_dim, latent_dim):
        super(FrontdoorSCM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        # Structural equations for state transitions
        self.linear_state = PyroModule[nn.Linear](state_dim + action_dim + latent_dim, state_dim)

        # Structural equations for rewards
        self.linear_reward = PyroModule[nn.Linear](state_dim + latent_dim, 1)

    def model(self, s, a, s_next=None, r=None):
        pyro.module("frontdoor_scm", self)
        batch_size = s.shape[0]
        with pyro.plate("data", size=batch_size):
            # Sample latent variable z per data point
            z = pyro.sample("z", dist.Normal(0., 1.).expand([batch_size, self.latent_dim]).to_event(1))
            # State transition
            s_a_z = torch.cat([s, a, z], dim=1)
            mu_s_next = self.linear_state(s_a_z)
            sigma_s_next = torch.ones_like(mu_s_next) * 0.1
            pyro.sample("s_next", dist.Normal(mu_s_next, sigma_s_next).to_event(1), obs=s_next)
            # Reward
            s_z = torch.cat([s_next, z], dim=1)
            mu_r = self.linear_reward(s_z)
            sigma_r = torch.ones_like(mu_r) * 0.1
            pyro.sample("r", dist.Normal(mu_r, sigma_r).to_event(1), obs=r)


    def guide(self, s, a, s_next=None, r=None):
        guide = AutoNormal(self.model)
        guide(s, a, s_next, r)
    def counterfactual(self, s, a, z):
        # Given s, a, and z, compute the counterfactual next state
        s_a_z = torch.cat([s, a, z], dim=1)
        mu_s_next = self.linear_state(s_a_z)
        return mu_s_next  # Return deterministic prediction for simplicity

    def predict_reward(self, s_next, z):
        s_z = torch.cat([s_next, z], dim=1)
        mu_r = self.linear_reward(s_z)
        return mu_r
