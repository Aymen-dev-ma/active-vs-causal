# main.py

import argparse
import torch
import numpy as np
from environment import GameEnvironment
from agent import CausalAgent, ActiveInferenceAgent
from utils import ReplayBuffer

def parse_arguments():
    parser = argparse.ArgumentParser(description='Causal vs Active Inference in RL')
    parser.add_argument('--agent', type=str, choices=['causal', 'active'], default='causal',
                        help='Type of agent to use: causal or active')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--save_model', type=str, default='trained_agent.pth', help='Path to save the trained model')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize environment
    env = GameEnvironment(games_no=1)
    state_dim = env.current_frame_all().view(-1).shape[0]
    action_space = [0, 1, 2, 3]  # Up, Down, Left, Right
    
    # Initialize agent
    if args.agent == 'causal':
        agent = CausalAgent(action_space=action_space, state_dim=state_dim)
    elif args.agent == 'active':
        agent = ActiveInferenceAgent(action_space=action_space, state_dim=state_dim)
    else:
        raise ValueError('Invalid agent type')
    
    # Initialize replay buffer
    buffer = ReplayBuffer(capacity=10000)
    
    # Training loop
    for episode in range(args.episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0  # Track the number of steps in the episode
    
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1  # Increment step count
    
            if len(buffer) > args.batch_size:
                batch = buffer.sample(args.batch_size)
                agent.update_model(batch)
    
            # Debug print to track progress
            print(f'Episode {episode + 1}, Step {step_count}, State: {state}, Action: {action}, Reward: {reward}, Done: {done}')
    
        print(f'Episode {episode + 1}/{args.episodes}, Total Reward: {total_reward}')
    
    # Save the trained model
    agent.save_model(args.save_model)

if __name__ == '__main__':
    main()
