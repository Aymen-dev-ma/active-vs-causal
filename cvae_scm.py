# cvae_scm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample

class CVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, condition_dim):
        super(CVAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim + condition_dim, 256)
        print(f"Initialized CVAE fc1 with input_dim={input_dim + condition_dim}, output_dim=256")
        self.fc21 = nn.Linear(256, latent_dim)  # Mean
        self.fc22 = nn.Linear(256, latent_dim)  # Log variance

        # Decoder
        self.fc3 = nn.Linear(latent_dim + condition_dim, 256)
        self.fc4 = nn.Linear(256, input_dim)

    def encode(self, x, c):
        concatenated = torch.cat([x, c], dim=1)
        h1 = F.relu(self.fc1(concatenated))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        concatenated = torch.cat([z, c], dim=1)
        h3 = F.relu(self.fc3(concatenated))
        return self.fc4(h3)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + KLD

class SCM(PyroModule):
    def __init__(self, state_dim, action_dim):
        super(SCM, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Define parameters as PyroSamples
        self.linear_state = PyroModule[nn.Linear](state_dim + action_dim, state_dim)
        self.linear_state.weight = PyroSample(dist.Normal(0., 1.).expand([state_dim, state_dim + action_dim]).to_event(2))
        self.linear_state.bias = PyroSample(dist.Normal(0., 10.).expand([state_dim]).to_event(1))

        self.linear_reward = PyroModule[nn.Linear](state_dim, 1)
        self.linear_reward.weight = PyroSample(dist.Normal(0., 1.).expand([1, state_dim]).to_event(2))
        self.linear_reward.bias = PyroSample(dist.Normal(0., 10.).expand([1]).to_event(1))

    def model(self, s, a, s_next=None, r=None):
        batch_size = s.size(0)
        # Sample global parameters
        sigma_s_next = pyro.sample("sigma_s_next", dist.HalfCauchy(scale=torch.ones(self.state_dim)).to_event(1))
        sigma_r = pyro.sample("sigma_r", dist.HalfCauchy(scale=1.0))

        # Sample parameters (weights and biases)
        weight_state = pyro.sample("linear_state.weight", dist.Normal(0., 1.).expand([self.state_dim, self.state_dim + self.action_dim]).to_event(2))
        bias_state = pyro.sample("linear_state.bias", dist.Normal(0., 10.).expand([self.state_dim]).to_event(1))
        weight_reward = pyro.sample("linear_reward.weight", dist.Normal(0., 1.).expand([1, self.state_dim]).to_event(2))
        bias_reward = pyro.sample("linear_reward.bias", dist.Normal(0., 10.).expand([1]).to_event(1))

        with pyro.plate("data", batch_size):
            sa = torch.cat([s, a], dim=1)
            mu_s_next = F.linear(sa, weight_state, bias_state)

            sigma_s_next_expanded = sigma_s_next.unsqueeze(0).expand_as(mu_s_next)

            if s_next is not None:
                pyro.sample("s_next", dist.Normal(mu_s_next, sigma_s_next_expanded).to_event(1), obs=s_next)
            else:
                s_next = pyro.sample("s_next", dist.Normal(mu_s_next, sigma_s_next_expanded).to_event(1))

            mu_r = F.linear(s_next, weight_reward, bias_reward)

            sigma_r_expanded = sigma_r.expand_as(mu_r)

            if r is not None:
                pyro.sample("r", dist.Normal(mu_r, sigma_r_expanded).to_event(1), obs=r)
            else:
                pyro.sample("r", dist.Normal(mu_r, sigma_r_expanded).to_event(1))

    def guide(self, s, a, s_next=None, r=None):
        # Variational distribution q
        # Define variational parameters for weights and biases
        weight_state_mean = pyro.param("weight_state_mean", torch.randn(self.state_dim, self.state_dim + self.action_dim))
        weight_state_std = pyro.param("weight_state_std", torch.ones(self.state_dim, self.state_dim + self.action_dim), constraint=dist.constraints.positive)
        bias_state_mean = pyro.param("bias_state_mean", torch.randn(self.state_dim))
        bias_state_std = pyro.param("bias_state_std", torch.ones(self.state_dim), constraint=dist.constraints.positive)

        weight_reward_mean = pyro.param("weight_reward_mean", torch.randn(1, self.state_dim))
        weight_reward_std = pyro.param("weight_reward_std", torch.ones(1, self.state_dim), constraint=dist.constraints.positive)
        bias_reward_mean = pyro.param("bias_reward_mean", torch.randn(1))
        bias_reward_std = pyro.param("bias_reward_std", torch.ones(1), constraint=dist.constraints.positive)

        # Sample parameters
        pyro.sample("linear_state.weight", dist.Normal(weight_state_mean, weight_state_std).to_event(2))
        pyro.sample("linear_state.bias", dist.Normal(bias_state_mean, bias_state_std).to_event(1))
        pyro.sample("linear_reward.weight", dist.Normal(weight_reward_mean, weight_reward_std).to_event(2))
        pyro.sample("linear_reward.bias", dist.Normal(bias_reward_mean, bias_reward_std).to_event(1))

        # Sample noise scales
        sigma_s_next_q = pyro.param("sigma_s_next_q", torch.ones(self.state_dim), constraint=dist.constraints.positive)
        pyro.sample("sigma_s_next", dist.Delta(sigma_s_next_q).to_event(1))

        sigma_r_q = pyro.param("sigma_r_q", torch.tensor(1.0), constraint=dist.constraints.positive)
        pyro.sample("sigma_r", dist.Delta(sigma_r_q))

    def counterfactual(self, s, a):
        try:
            predictive = pyro.infer.Predictive(self.model, guide=self.guide, num_samples=100, return_sites=("s_next",))
            samples = predictive(s, a)
            s_next_samples = samples["s_next"]  # Shape: [num_samples, batch_size, state_dim]
            s_next_mean = s_next_samples.mean(dim=0)  # Shape: [batch_size, state_dim]
            return s_next_mean  # Return [batch_size, state_dim]
        except Exception as e:
            print(f"Exception in counterfactual: {e}")
            return None

    def predict_reward(self, s):
        with torch.no_grad():
            mu_r = self.linear_reward(s)  # s: [batch_size, state_dim], mu_r: [batch_size, 1]
            return mu_r.item()

    def update(self, s, a, s_next, r):
        # Update SCM parameters using stochastic variational inference
        pyro.clear_param_store()
        optimizer = pyro.optim.Adam({"lr": 1e-3})
        svi = pyro.infer.SVI(self.model, self.guide, optimizer, loss=pyro.infer.Trace_ELBO())

        num_iterations = 100  # Adjust as needed
        for i in range(num_iterations):
            loss = svi.step(s, a, s_next, r)

        # After training, update the linear_reward and linear_state parameters
        with torch.no_grad():
            self.linear_reward.weight.copy_(pyro.param("weight_reward_mean"))
            self.linear_reward.bias.copy_(pyro.param("bias_reward_mean"))
            self.linear_state.weight.copy_(pyro.param("weight_state_mean"))
            self.linear_state.bias.copy_(pyro.param("bias_state_mean"))

    def state_dict(self):
        # Return Pyro parameters
        return pyro.get_param_store().get_state()

    def load_state_dict(self, state_dict):
        # Load Pyro parameters
        pyro.get_param_store().set_state(state_dict)

    def get_pyro_param_store(self):
        return pyro.get_param_store().get_state()
