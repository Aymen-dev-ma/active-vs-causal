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



# FrontdoorAgent: Integrating CVAE, SCM, and MCTS for Causal Decision-Making

## Introduction

The **FrontdoorAgent** is an advanced reinforcement learning agent designed to make smarter decisions by integrating causal inference techniques into its learning and planning processes. Specifically, it leverages the **front-door criterion** from causal inference to adjust for unobserved confounders, allowing for more accurate estimation of causal effects in its environment.

This README provides a comprehensive explanation of how the FrontdoorAgent operates within a custom `GameEnvironment`, detailing how it uses the **Conditional Variational Autoencoder (CVAE)**, **Structural Causal Model (SCM)**, and **Monte Carlo Tree Search (MCTS)** in unison.

---

## Table of Contents

- [Introduction](#introduction)
- [Background Concepts](#background-concepts)
  - [Reinforcement Learning (RL)](#1-reinforcement-learning-rl)
  - [Causal Inference](#2-causal-inference)
  - [Front-Door Criterion](#3-front-door-criterion)
- [Tools Used](#tools-used)
- [The Game Environment](#the-game-environment)
  - [Environment Overview](#1-environment-overview)
  - [Dataset Integration](#2-dataset-integration)
- [FrontdoorAgent's Operation](#frontdooragents-operation)
  - [Modeling Unobserved Confounders with CVAE](#1-modeling-unobserved-confounders-with-cvae)
  - [Capturing Causal Relationships with SCM](#2-capturing-causal-relationships-with-scm)
  - [Planning with Monte Carlo Tree Search (MCTS)](#3-planning-with-monte-carlo-tree-search-mcts)
- [Linking the Components Together](#linking-the-components-together)
  - [Detailed Step-by-Step Interaction](#detailed-step-by-step-interaction)
  - [Visualization of the Interaction Flow](#visualization-of-the-interaction-flow)
  - [Example Scenario](#example-scenario)
- [Conclusion](#conclusion)
- [Final Remarks](#final-remarks)
- [Further Reading](#further-reading)

---

## Background Concepts

### 1. Reinforcement Learning (RL)

**Reinforcement Learning** is a type of machine learning where an agent learns to make decisions by interacting with an environment to maximize cumulative rewards.

- **Agent**: Learner and decision-maker.
- **Environment**: The external system the agent interacts with.
- **State (s)**: A representation of the current situation.
- **Action (a)**: A choice made by the agent.
- **Reward (r)**: Feedback from the environment.
- **Policy (π)**: Strategy that the agent employs to determine actions based on states.

### 2. Causal Inference

**Causal Inference** is the process of determining cause-and-effect relationships between variables.

- **Confounders**: Variables that affect both the treatment (action) and the outcome (reward), potentially biasing estimates of causal effects.
- **Front-Door Criterion**: A method for estimating causal effects in the presence of unobserved confounders by adjusting for an intermediate variable.

### 3. Front-Door Criterion

The **front-door criterion** allows for causal effect estimation even when there are unobserved confounders, provided certain conditions are met.

- **Key Idea**: Use an intermediate variable that is influenced by the treatment and influences the outcome but is not affected by unobserved confounders.
- **Adjustment**: By conditioning on this intermediate variable, we can block the backdoor paths from the treatment to the outcome.

---

## Tools Used

- **PyTorch**: An open-source machine learning library for tensor computation and deep learning.
- **Pyro**: A probabilistic programming language built on PyTorch for flexible and expressive specification of probabilistic models.
- **Conditional Variational Autoencoder (CVAE)**: A type of variational autoencoder that models the conditional distribution of data given some condition, used to learn latent representations.
- **Structural Causal Model (SCM)**: A mathematical model representing causal relationships using structural equations, enabling counterfactual reasoning.
- **Monte Carlo Tree Search (MCTS)**: A heuristic search algorithm for decision processes, balancing exploration and exploitation.

---

## The Game Environment

### 1. Environment Overview

The `GameEnvironment` simulates a simple grid-based game where an agent moves in a multidimensional state space.

- **State Space (`self.current_s`)**:
  - Each state is a vector of size `s_dim` (number of latent dimensions + 1 for reward).
  - The state vector includes elements influenced by the agent's actions and others influenced by latent factors.

- **Action Space**:
  - Four possible actions:
    - `0`: Move up (increment the first state dimension).
    - `1`: Move down (decrement the first state dimension).
    - `2`: Move left (decrement the second state dimension).
    - `3`: Move right (increment the second state dimension).

- **Reward Function**:
  - The reward is the negative sum of the absolute values of the state vector:
    \[
    \text{reward} = -\sum_{i} |s_i|
    \]
  - Incentivizes the agent to minimize the absolute values of the state dimensions.

- **Episode Termination**:
  - An episode ends after a maximum number of steps (`max_steps_per_episode`, default 100).

### 2. Dataset Integration

- The environment loads a dataset containing images and corresponding latent variables.
- While images are loaded, they are not directly used in the agent's decision-making.
- Latent variables represent factors like shape, scale, orientation, and position.

---

## FrontdoorAgent's Operation

### Overview

The **FrontdoorAgent** aims to make optimal decisions by leveraging causal inference techniques to adjust for unobserved confounders—latent variables in the environment that the agent doesn't directly observe but that affect state transitions and rewards. The agent uses:

- A **CVAE** to model the distribution of latent variables given observed states and actions.
- An **SCM** to capture the causal relationships between variables.
- **MCTS** for planning and selecting the best actions based on causal simulations.

### 1. Modeling Unobserved Confounders with CVAE

#### a. Purpose in This Environment

- Captures latent variables representing unobserved confounders influencing state transitions and rewards.
- Learns a latent representation (`z`) of these unobserved factors based on observed states and actions.

#### b. How the CVAE Works Here

- **Encoder**:
  - Takes the current state `s` as input.
  - Produces the mean `μ` and log-variance `logσ²` of the latent variable `z`.
- **Decoder**:
  - Takes the sampled latent variable `z` and the action `a`.
  - Attempts to reconstruct the state or predict the next state.
- **Training**:
  - The CVAE is trained during interaction with the environment, using the agent's collected experiences.

### 2. Capturing Causal Relationships with SCM

#### a. Purpose in This Environment

- Models how actions and latent variables influence state transitions and rewards.
- Enables counterfactual reasoning to predict the effect of different actions.

#### b. How the SCM Works Here

- **State Transition Model**:
  - Predicts the next state `s'` based on the current state `s`, action `a`, and latent variable `z`.
    \[
    s' = f_{\text{state}}(s, a, z)
    \]
- **Reward Model**:
  - Predicts the reward `r` based on the next state `s'` and latent variable `z`.
    \[
    r = f_{\text{reward}}(s', z)
    \]
- **Learning**:
  - The SCM parameters are learned using **Stochastic Variational Inference (SVI)** with Pyro.

### 3. Planning with Monte Carlo Tree Search (MCTS)

#### a. Purpose in This Environment

- MCTS allows the agent to plan ahead by simulating possible future action sequences.
- Incorporating the CVAE and SCM into MCTS helps the agent consider the impact of unobserved confounders in its planning.

#### b. How MCTS Works Here

- **Selection**:
  - Starting from the root node (current state), select child nodes that maximize an exploration-exploitation trade-off.
- **Expansion**:
  - Add new child nodes corresponding to untried actions.
- **Simulation**:
  - Use the SCM and sampled latent variables (`z`) to simulate state transitions and rewards.
- **Backpropagation**:
  - Update the values and visit counts of nodes based on simulation results.
- **Action Selection**:
  - Select the action with the highest value or visit count from the root node after simulations.

---

## Linking the Components Together

### Overview

The **FrontdoorAgent** uses the CVAE, SCM, and MCTS in a tightly integrated manner to make informed decisions that account for unobserved confounders.

### Detailed Step-by-Step Interaction

#### **Step 1: Observing the Current State**

- The agent receives the current state vector `s` from the environment.

#### **Step 2: Inferring Latent Variables with CVAE**

- **CVAE Encoding**:
  - Inputs `s` into the CVAE's encoder.
  - Outputs `μ` and `σ²` for the latent variable `z`.

- **Sampling Latent Variables**:
  - Samples `z` from the distribution `q(z | s)`.

**Interaction**:

- The CVAE provides `z` to adjust for unobserved confounders using the front-door criterion.

#### **Step 3: Planning with MCTS Using SCM and Latent Variables**

**MCTS Simulations**:

1. **Selection**:
   - Traverses the tree using a selection policy.

2. **Expansion**:
   - Expands nodes by adding new actions.

3. **Simulation**:
   - **SCM Prediction**:
     - Predicts `s' = f_{\text{state}}(s, a, z)`.
     - Predicts `r = f_{\text{reward}}(s', z)`.

4. **Backpropagation**:
   - Updates values and visit counts.

**Interaction**:

- MCTS relies on SCM and `z` for simulations.
- SCM uses `z` from CVAE to model causal effects accurately.

#### **Step 4: Selecting the Best Action**

- Selects the action `a` from the root node with the highest value or visit count.

**Interaction**:

- MCTS provides an action informed by causal simulations accounting for unobserved confounders.

#### **Step 5: Executing the Action and Observing the Outcome**

- Executes action `a` in the environment.
- Receives new state `s'` and reward `r`.

#### **Step 6: Updating Models (Learning)**

- **Experience Storage**:
  - Stores `(s, a, r, s')` in a replay buffer.

- **Updating the CVAE**:
  - Trains using experiences to refine latent variable approximation.

- **Updating the SCM**:
  - Updates parameters using SVI.

**Interaction**:

- CVAE and SCM are updated based on experiences to improve future predictions.

#### **Step 7: Repeat**

- The process repeats from **Step 1** with the new state.

### Visualization of the Interaction Flow

```mermaid
flowchart TD
    A[Current State s] --> B[CVAE Encoder]
    B --> C[Latent Variable z]
    C --> D[SCM]
    D --> E[MCTS Simulations]
    E --> F[Best Action a]
    F --> G[Execute Action a in Environment]
    G --> H[New State s', Reward r]
    H --> I[Experience (s, a, r, s')]
    I --> J[Update CVAE and SCM]
    J --> K[Next Iteration]
