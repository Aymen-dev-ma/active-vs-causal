# main.py

import argparse
import torch
import numpy as np
from environment import GameEnvironment
from agent import CausalAgent, ActiveInferenceAgent
from utils import ReplayBuffer

def parse_arguments():
    parser = argparse.ArgumentParser(description='Active Inference vs Causal Agent')
    parser.add_argument('--agent', type=str, choices=['causal', 'active', 'both'], default='both',
                        help='Type of agent to use: causal, active, or both')
    parser.add_argument('--episodes', type=int, default=10, help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--save_model', type=str, default='trained_agent.pth', help='Path to save the trained model')
    return parser.parse_args()

def train_agent(agent, env, buffer, args):
    total_rewards = []
    for episode in range(args.episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        step = 0

        print(f"Starting Episode {episode + 1}")

        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1

            # Print step information
            print(f"Episode {episode + 1}, Step {step}, Action: {action}, Reward: {reward:.2f}, Done: {done}")

            # Update model if we have enough samples
            if len(buffer) >= args.batch_size:
                batch = buffer.sample(args.batch_size)
                agent.update_model(batch)

        total_rewards.append(total_reward)
        print(f'Episode {episode + 1}/{args.episodes}, Total Reward: {total_reward}, Steps: {step}')
    return total_rewards

def main():
    args = parse_arguments()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize environment
    env = GameEnvironment(games_no=1, max_steps_per_episode=100)
    state_dim = env.current_s.shape[1]  # Get the state dimension from the environment
    action_space = [0, 1, 2, 3]  # Up, Down, Left, Right
    
    if args.agent == 'both':
        agents = {
            'causal': CausalAgent(action_space=action_space, state_dim=state_dim),
            'active': ActiveInferenceAgent(action_space=action_space, state_dim=state_dim)
        }
        results = {}
        for agent_name, agent in agents.items():
            print(f"\nTraining {agent_name.capitalize()} Agent")
            buffer = ReplayBuffer(capacity=10000)
            total_rewards = train_agent(agent, env, buffer, args)
            results[agent_name] = total_rewards
            env.reset()
        # Compare results
        print("\nComparison of Agents:")
        for agent_name in agents.keys():
            avg_reward = np.mean(results[agent_name])
            print(f"{agent_name.capitalize()} Agent Average Reward: {avg_reward}")
    else:
        # Initialize agent
        if args.agent == 'causal':
            agent = CausalAgent(action_space=action_space, state_dim=state_dim)
        elif args.agent == 'active':
            agent = ActiveInferenceAgent(action_space=action_space, state_dim=state_dim)
        else:
            raise ValueError('Invalid agent type')
        
        # Initialize replay buffer
        buffer = ReplayBuffer(capacity=10000)
        
        # Train agent
        total_rewards = train_agent(agent, env, buffer, args)
        avg_reward = np.mean(total_rewards)
        print(f"\n{args.agent.capitalize()} Agent Average Reward: {avg_reward}")
        
        # Save the trained model
        agent.save_model(args.save_model)

if __name__ == '__main__':
    main()
