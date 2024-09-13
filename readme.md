# Active Inference vs. Causal Agent: A Comparative Study in Reinforcement Learning

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Methodology](#methodology)
  - [Environment](#environment)
  - [Agents](#agents)
    - [Causal Agent](#causal-agent)
    - [Active Inference Agent](#active-inference-agent)
  - [Models](#models)
    - [Conditional Variational Autoencoder (CVAE)](#conditional-variational-autoencoder-cvae)
    - [Structural Causal Model (SCM)](#structural-causal-model-scm)
- [Implementation Details](#implementation-details)
  - [Project Structure](#project-structure)
  - [Dependencies](#dependencies)
  - [Setup and Installation](#setup-and-installation)
  - [Running the Experiments](#running-the-experiments)
    - [Training the Agents](#training-the-agents)
    - [Comparing Agent Performance](#comparing-agent-performance)
  - [Expected Results](#expected-results)
  - [Technical Details](#technical-details)
    - [Mathematical Formulation](#mathematical-formulation)
    - [Inference Techniques](#inference-techniques)
- [References](#references)

## Introduction
This project is a comparative study of two reinforcement learning agents: a Causal Agent and an Active Inference Agent. The goal is to evaluate the performance of these agents in a controlled environment and analyze their decision-making processes. The study leverages advanced probabilistic models, such as Conditional Variational Autoencoders (CVAEs) and Structural Causal Models (SCMs), to enable the agents to learn and reason about the environment.

This README provides a comprehensive guide to the project, including detailed technical explanations suitable for a master's thesis.

## Project Overview
The project involves:

- Implementing a game environment based on the dSprites dataset, which provides a controlled setting for testing the agents.
- Developing two agents:
  - **Causal Agent**: Uses SCMs to perform counterfactual reasoning and make decisions based on predicted rewards.
  - **Active Inference Agent**: Employs the principles of active inference, minimizing expected free energy to select actions.
- Comparing the performance of the agents over a series of episodes to evaluate their effectiveness.

## Methodology

### Environment
The `GameEnvironment` simulates a simple grid-based world where the agents can perform actions such as moving up, down, left, or right. The state space is derived from the latent factors of the dSprites dataset, providing a high-dimensional and structured state representation.

- **State Representation**: Each state is a vector containing latent variables like shape, scale, orientation, and position.
- **Action Space**: Discrete actions representing movement in the grid.
- **Reward Function**: Negative sum of the absolute values of the state variables, encouraging the agent to reach a state close to zero.

### Agents

#### Causal Agent
The Causal Agent utilizes a Structural Causal Model to perform counterfactual reasoning. It predicts the potential outcomes of different actions and selects the one with the highest expected reward.

- **Counterfactual Reasoning**: Uses SCM to simulate the effect of actions on the environment.
- **Decision Making**: Chooses actions based on predicted rewards from the SCM.
- **Learning Mechanism**: Updates the SCM parameters using stochastic variational inference.

#### Active Inference Agent
The Active Inference Agent operates based on the principles of active inference, aiming to minimize the expected free energy (EFE).

- **Expected Free Energy (EFE)**: Measures the divergence between predicted states and prior preferences.
- **Decision Making**: Selects actions that minimize EFE, effectively reducing uncertainty.
- **Learning Mechanism**: Uses a CVAE to model the generative process of the environment.

### Models

#### Conditional Variational Autoencoder (CVAE)
The CVAE is used by both agents to model the conditional distribution of the next state given the current state and action.

- **Encoder**: Maps input state-action pairs to a latent space.
- **Decoder**: Reconstructs the next state from the latent representation and action.
- **Loss Function**: Combines reconstruction loss and Kullback-Leibler divergence to enforce a structured latent space.

#### Structural Causal Model (SCM)
The SCM is a probabilistic model that captures the causal relationships between variables.

- **Causal Graph**: Represents the dependencies between state variables, actions, and rewards.
- **Parameter Learning**: Uses variational inference to estimate the distributions of the model parameters.
- **Counterfactuals**: Enables the agent to predict what would happen under different actions.

## Implementation Details

### Project Structure
```bash
active_vs_causal/
├── main.py                   # Main script to run and compare the agents
├── agent.py                  # Implementation of the Causal and Active Inference agents
├── environment.py            # Definition of the GameEnvironment class
├── cvae_scm.py               # Implementation of CVAE and SCM models
├── utils.py                  # Utility functions (e.g., ReplayBuffer)
├── requirements.txt          # List of required Python packages
└── README.md                 # Project documentation
# Active Inference vs. Causal Agent: A Comparative Study in Reinforcement Learning

## Dependencies
- Python 3.7+
- PyTorch (>=1.7.0)
- NumPy
- Pyro (>=1.6.0)

## Setup and Installation

### Clone the repository:
```bash
git clone https://github.com/evvco/active_vs_causal.git
cd active_vs_causal
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run the experiments:
```bash 
python main.py --agent both --episodes 10
```
Comparing Agent Performance
When running with --agent both, the script will:

Train both agents sequentially.
Store the total rewards per episode.
Output the average reward for each agent at the end.

## Technical Details

### Mathematical Formulation

#### Conditional Variational Autoencoder (CVAE)

Objective Function:

log p_θ(x | c) = - ELBO = - E_q_ϕ(z | x, c) [log p_θ(x | z, c)] + KL(q_ϕ(z | x, c) || p(z | c))



Where:
- **x**: Data variable (state).
- **c**: Condition variable (action).
- **z**: Latent variable.
- **θ, ϕ**: Parameters of the generative and inference networks, respectively.

#### Structural Causal Model (SCM)

Model Structure:

s_next = f_s(s, a, ε_s) r = f_r(s_next, ε_r)



Where:
- **s**: Current state.
- **a**: Action.
- **s_next**: Next state.
- **r**: Reward.
- **ε_s, ε_r**: Exogenous noise variables.

### Inference Techniques

- **Variational Inference**: Used in both the CVAE and SCM to approximate intractable posterior distributions.
- **Stochastic Variational Inference**: Optimizes the Evidence Lower Bound (ELBO) using stochastic gradient descent.
- **Predictive Sampling**: The `pyro.infer.Predictive` class is used to draw samples from the posterior predictive distribution, facilitating counterfactual reasoning.
