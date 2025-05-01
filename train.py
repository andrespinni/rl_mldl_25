"""Train an RL agent on the OpenAI Gym Hopper environment using
REINFORCE and Actor-critic algorithms
"""
import argparse

import torch
import gym

from env.custom_hopper import *
from agent import Agent, Policy

import wandb
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', default=100000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=10000, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--name', default='hopper-train_noName', type=str, help='Scegliere nome')

    return parser.parse_args()

args = parse_args()


def main():
    
    os.makedirs(args.name, exist_ok=True)
    out_file_name = f"{args.name}/output_train.txt"
    out_file = open(out_file_name, "w")

    env = gym.make('CustomHopper-source-v0')
    # env = gym.make('CustomHopper-target-v0')

    wandb.init(
        project="ML_project",
        name=f"{args.name}_train",
        entity="andrea-gaudino02-politecnico-di-torino",
        config={
            "env": "CustomHopper-source-v0",
            "episodes": args.episodes,
            "print_every": args.print_every,
            "device": args.device,
            "algorithm": "REINFORCE"  # Puoi personalizzare
        }
    )

    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())
    
    out_file.write(f"Action space: {env.action_space}\n")
    out_file.write(f"State space: {env.observation_space}\n")
    out_file.write(f"Dynamic parameters: {env.get_parameters()}\n")

    # Training
    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device=args.device)

    # TASK 2 and 3: interleave data collection to policy updates
    for episode in range(args.episodes):
        done = False
        train_reward = 0
        state = env.reset()  # Reset the environment and observe the initial state

        while not done:  # Loop until the episode is over
            action, action_probabilities = agent.get_action(state)
            previous_state = state
            state, reward, done, info = env.step(action.detach().cpu().numpy())

            agent.store_outcome(previous_state, state, action_probabilities, reward, done)
            train_reward += reward

        agent.update_policy()

        wandb.log({"episode": episode + 1, "train_reward": train_reward})

        if (episode + 1) % args.print_every == 0:
            print('Training episode:', episode + 1)
            print('Episode return:', train_reward)
            
            out_file.write(f"Training episode: {episode+1}\n")
            out_file.write(f"Episode return: {train_reward}\n")

    torch.save(agent.policy.state_dict(), f"{args.name}/model.mdl")

    artifact = wandb.Artifact('model', type='model')
    artifact.add_file(f'{args.name}/model.mdl')
    wandb.log_artifact(artifact)

    wandb.finish()
    out_file.close()


if __name__ == '__main__':
    main()
