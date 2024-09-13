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
        self.fc21 = nn.Linear(256, latent_dim)  # Mean
        self.fc22 = nn.Linear(256, latent_dim)  # Log variance

        # Decoder
        self.fc3 = nn.Linear(latent_dim + condition_dim, 256)
        self.fc4 = nn.Linear(256, input_dim)

    def encode(self, x, c):
        h1 = F.relu(self.fc1(torch.cat([x, c], dim=1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        h3 = F.relu(self.fc3(torch.cat([z, c], dim=1)))
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

        # Prior distributions for the parameters
        self.linear_state = PyroModule[nn.Linear](state_dim + action_dim, state_dim)
        self.linear_state.weight = PyroSample(dist.Normal(0., 1.).expand([state_dim, state_dim + action_dim]).to_event(2))
        self.linear_state.bias = PyroSample(dist.Normal(0., 10.).expand([state_dim]).to_event(1))

        self.linear_reward = PyroModule[nn.Linear](state_dim, 1)
        self.linear_reward.weight = PyroSample(dist.Normal(0., 1.).expand([1, state_dim]).to_event(2))
        self.linear_reward.bias = PyroSample(dist.Normal(0., 10.).expand([1]).to_event(1))

    def model(self, s, a, s_next=None, r=None):
        # Generative model (probabilistic structural equations)
        batch_size = s.size(0)
        with pyro.plate("data", batch_size):
            # Sample parameters (weights and biases)
            weight_state = self.linear_state.weight
            bias_state = self.linear_state.bias
            weight_reward = self.linear_reward.weight
            bias_reward = self.linear_reward.bias

            # Compute mean of the next state
            sa = torch.cat([s, a], dim=1)
            mu_s_next = F.linear(sa, weight_state, bias_state)

            # Sample next state
            sigma_s_next = pyro.sample("sigma_s_next", dist.HalfCauchy(scale=1.0))
            with pyro.poutine.scale(scale=5.0):  # Scaling the loss for s_next
                pyro.sample("s_next", dist.Normal(mu_s_next, sigma_s_next).to_event(1), obs=s_next)

            # Compute mean of the reward
            mu_r = F.linear(s_next, weight_reward, bias_reward)

            # Sample reward
            sigma_r = pyro.sample("sigma_r", dist.HalfCauchy(scale=1.0))
            pyro.sample("r", dist.Normal(mu_r, sigma_r).to_event(1), obs=r)

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
        sigma_s_next = pyro.param("sigma_s_next_q", torch.tensor(1.0), constraint=dist.constraints.positive)
        pyro.sample("sigma_s_next", dist.Delta(sigma_s_next))
        sigma_r = pyro.param("sigma_r_q", torch.tensor(1.0), constraint=dist.constraints.positive)
        pyro.sample("sigma_r", dist.Delta(sigma_r))

    def counterfactual(self, s, a):
        # Counterfactual reasoning using the posterior predictive distribution
        # s: [1, state_dim], a: [1, action_dim]
        predictive = pyro.infer.Predictive(self.model, guide=self.guide, num_samples=100)
        samples = predictive(s.repeat(100, 1), a.repeat(100, 1))
        s_next_samples = samples["s_next"]
        s_next_mean = s_next_samples.mean(dim=0)
        return s_next_mean.unsqueeze(0)  # Return [1, state_dim]

    def predict_reward(self, s):
        # Predict reward from state
        with torch.no_grad():
            weight_reward_mean = pyro.param("weight_reward_mean")
            bias_reward_mean = pyro.param("bias_reward_mean")
            mu_r = F.linear(s, weight_reward_mean, bias_reward_mean)
            return mu_r.item()

    def update(self, s, a, s_next, r):
        # Update SCM parameters using stochastic variational inference
        pyro.clear_param_store()
        optimizer = pyro.optim.Adam({"lr": 1e-3})
        svi = pyro.infer.SVI(self.model, self.guide, optimizer, loss=pyro.infer.Trace_ELBO())

        num_iterations = 100  # Adjust as needed
        for i in range(num_iterations):
            loss = svi.step(s, a, s_next, r)

    def state_dict(self):
        # Return Pyro parameters
        return pyro.get_param_store().get_state()

    def load_state_dict(self, state_dict):
        # Load Pyro parameters
        pyro.get_param_store().set_state(state_dict)

    def get_pyro_param_store(self):
        return pyro.get_param_store().get_state()
