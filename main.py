# main.py

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from environment import GameEnvironment
from agent import (
    CausalUCTAgent,
    CausalAgent,
    ActiveInferenceAgent,
    CEVAEAgent,
    IPSAgent,
)
from utils import ReplayBuffer

def parse_arguments():
    parser = argparse.ArgumentParser(description='Agent Comparison')
    parser.add_argument('--agent', type=str,
                        choices=['uct', 'causal', 'active', 'cevae', 'ips', 'all'],
                        default='all',
                        help='Type of agent to use: uct, causal, active, cevae, ips, or all')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--save_model', type=str, default='trained_agent.pth',
                        help='Path to save the trained model')
    return parser.parse_args()

def train_agent(agent, env, buffer, args, agent_name):
    step_rewards = []
    for episode in range(args.episodes):
        state = env.reset()
        done = False
        step = 0

        print(f"Starting Episode {episode + 1}")

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            step_rewards.append(reward)
            step += 1

            if len(buffer) >= args.batch_size:
                batch = buffer.sample(args.batch_size)
                agent.update_model(batch)

        print(f'Episode {episode + 1}/{args.episodes}, Steps: {step}')
    return step_rewards

def plot_rewards(results):
    plt.figure(figsize=(10, 6))
    for agent_name, rewards in results.items():
        plt.plot(rewards, label=f'{agent_name.capitalize()} Agent')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Training Rewards per Step')
    plt.legend()
    plt.show()

def main():
    args = parse_arguments()

    torch.manual_seed(42)
    np.random.seed(42)

    env = GameEnvironment(games_no=1, max_steps_per_episode=100)
    state_dim = env.current_s.shape[1]
    action_space = [0, 1, 2, 3]

    if args.agent == 'all':
        agents = {
            'uct': CausalUCTAgent(action_space=action_space, state_dim=state_dim),
            'causal': CausalAgent(action_space=action_space, state_dim=state_dim),
            'active': ActiveInferenceAgent(action_space=action_space, state_dim=state_dim),
            'cevae': CEVAEAgent(action_space=action_space, state_dim=state_dim),
            'ips': IPSAgent(action_space=action_space, state_dim=state_dim),
        }
        results = {}
        for agent_name, agent in agents.items():
            print(f"\nTraining {agent_name.capitalize()} Agent")
            buffer = ReplayBuffer(capacity=10000)
            step_rewards = train_agent(agent, env, buffer, args, agent_name)
            results[agent_name] = step_rewards
            env.reset()
        print("\nComparison of Agents:")
        for agent_name in agents.keys():
            avg_reward = np.mean(results[agent_name])
            print(f"{agent_name.capitalize()} Agent Average Reward: {avg_reward}")
        
        # Plot the rewards
        plot_rewards(results)
    else:
        if args.agent == 'uct':
            agent = CausalUCTAgent(action_space=action_space, state_dim=state_dim)
        elif args.agent == 'causal':
            agent = CausalAgent(action_space=action_space, state_dim=state_dim)
        elif args.agent == 'active':
            agent = ActiveInferenceAgent(action_space=action_space, state_dim=state_dim)
        elif args.agent == 'cevae':
            agent = CEVAEAgent(action_space=action_space, state_dim=state_dim)
        elif args.agent == 'ips':
            agent = IPSAgent(action_space=action_space, state_dim=state_dim)
        else:
            raise ValueError('Invalid agent type')

        buffer = ReplayBuffer(capacity=10000)
        step_rewards = train_agent(agent, env, buffer, args, args.agent)
        avg_reward = np.mean(step_rewards)
        print(f"\n{args.agent.capitalize()} Agent Average Reward: {avg_reward}")

        agent.save_model(args.save_model)
        
        # Plot the rewards for the single agent
        plot_rewards({args.agent: step_rewards})

if __name__ == '__main__':
    main()
