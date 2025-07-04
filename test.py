"""Test an RL agent on the OpenAI Gym Hopper environment"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy

import wandb
import os

import time

def parse_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', default=None, type=str, help='Model path')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--episodes', default=1000, type=int, help='Number of test episodes')
    parser.add_argument('--name', required=True, type=str, help='Scegliere nome')
    return parser.parse_args()

args = parse_args()
seeds = [100, 200, 300, 400, 500]

def main(seed):
    
    os.makedirs(args.name, exist_ok=True)
    out_file_name = f"{args.name}/output_test_seed_{seed}.txt"
    out_file = open(out_file_name, "w")  # Corretto: aggiunta modalità "w"

    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')
    
    env.seed(seed)

    wandb.init(
        project="ML_project",
        name=f"{args.name}_test_seed_{seed}",
        entity="andrea-gaudino02-politecnico-di-torino",
        config={
            "env": "CustomHopper-source-v0",
            #"model_path": args.model,
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
    
    model_path = f"{args.name}/model.mdl"

    policy = Policy(observation_space_dim, action_space_dim)
    policy.load_state_dict(torch.load(model_path), strict=True)

    agent = Agent(policy, device=args.device)

    for episode in range(args.episodes):

        start = time.time()

        done = False
        test_reward = 0
        state = env.reset()

        while not done:
            action, _ = agent.get_action(state, evaluation=True)
            state, reward, done, info = env.step(action.detach().cpu().numpy())

            if args.render:
                env.render()

            test_reward += reward

        end = time.time()

        tempo_test = end - start
        
        print(f"Episode: {episode+1} | Return: {test_reward}")
        out_file.write(f"Episode: {episode+1} | Return: {test_reward}\n")
        wandb.log({"episode": episode+1, "test_reward": test_reward, "tempo_test": tempo_test}, step=episode+1)

    out_file.close()
    wandb.finish()

if __name__ == '__main__':
    for seed in seeds:
        main(seed)
