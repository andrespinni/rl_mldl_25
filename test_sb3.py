"""Test an RL agent trained with Stable-Baselines3 on the CustomHopper environment"""
import argparse
import os
import time

import gym
import wandb
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Importa la tua env personalizzata
from env.custom_hopper import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to the saved model (.zip)')
    parser.add_argument('--episodes', type=int, default=100, help='Number of test episodes')
    parser.add_argument('--render', default=False, action='store_true', help='Render the simulator')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on [cpu, cuda]')
    parser.add_argument('--name', type=str, default='sb3-test', help='Name of the test run')
    parser.add_argument("--mod_test", default="source",type=str)

    return parser.parse_args()

def main():
    args = parse_args()
    mod_test=args.mod_test

    # Output file
    os.makedirs(args.name, exist_ok=True)
    out_file_path = os.path.join(args.name, f"output_test_{mod_test}.txt")
    out_file = open(out_file_path, "w")

    # Inizializza ambiente
    env = gym.make(f'CustomHopper-{mod_test}-v0')

    # Log iniziale
    print("Action space:", env.action_space)
    print("State space:", env.observation_space)
    print("Dynamics parameters:", env.get_parameters())
    
    
    out_file.write(f"Model testato con {mod_test}\n")

    out_file.write(f"Action space: {env.action_space}\n")
    out_file.write(f"State space: {env.observation_space}\n")
    out_file.write(f"Dynamics parameters: {env.get_parameters()}\n")

    # Inizializza W&B
    wandb.init(
        project="PPO",
        name=f"{args.name}_test_{mod_test}",
        entity="andrea-gaudino02-politecnico-di-torino",
        config={
            "model_path": args.model,
            "episodes": args.episodes,
            "render": args.render,
            "device": args.device,
            "env": "CustomHopper-target-v0"
        }
    )

    # Carica modello SB3
    model = PPO.load(args.model, device=args.device)


    for episode in range(1, args.episodes + 1):
        done = False
        state = env.reset()
        test_reward = 0
        start_time = time.time()

        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, done, _ = env.step(action)

            if args.render:
                env.render()

            test_reward += reward

        duration = time.time() - start_time

        print(f"Episode {episode} | Return: {test_reward:.2f} | Duration: {duration:.2f}s")
        out_file.write(f"Episode {episode} | Return: {test_reward:.2f} | Duration: {duration:.2f}s\n")
        wandb.log({
            "episode": episode,
            "test_reward": test_reward,
            "test_duration": duration
        }, step=episode)


    out_file.close()
    wandb.finish()

if __name__ == '__main__':
    main()
