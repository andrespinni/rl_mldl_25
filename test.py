"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy

import wandb
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=1000, type=int, help='Number of test episodes')
    parser.add_argument('--name', default='hopper-test_noName', type=str, help='Scegliere nome')
    return parser.parse_args()

args = parse_args()

def main():
    out_file_name = f"{args.name}/output_test.txt"
    out_file = open(out_file_name, "w")  # Corretto: aggiunta modalità "w"

    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    wandb.init(
        project=args.name,
        config={
            "env": "CustomHopper-source-v0",
            "model_path": args.model,
            "device": args.device,
            "episodes": args.episodes
        }
    )

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())
    print("Rendering: ", args.render)
    
    out_file.write(f"Action space: {env.action_space}\n")
    out_file.write(f"State space: {env.observation_space}\n")
    out_file.write(f"Dynamic parameters: {env.get_parameters()}\n")
    out_file.write(f"Rendering: {args.render}\n")

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    policy.load_state_dict(torch.load(args.model), strict=True)

    agent = Agent(policy, device=args.device)

    for episode in range(args.episodes):
        done = False
        test_reward = 0
        state = env.reset()

        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            state, reward, done, info = env.step(action.detach().cpu().numpy())

            if args.render:
                env.render()

            test_reward += reward

        print(f"Episode: {episode} | Return: {test_reward}")
        out_file.write(f"Episode: {episode} | Return: {test_reward}\n")
        wandb.log({"episode": episode, "test_reward": test_reward})

    out_file.close()
    wandb.finish()

if __name__ == '__main__':
    main()
