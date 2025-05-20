import gym
import os
from stable_baselines3 import PPO
import argparse
import optuna
from stable_baselines3 import PPO
from env.custom_hopper import *
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', default=1_000_000, type=int, help='Number of training episodes')
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--name', default='hopper-train_noName', type=str, help='Scegliere nome')
    parser.add_argument("--mod_train", default="source",type=str)
    
    return parser.parse_args()

args = parse_args()

mod_train=args.mod_train

def optimize_ppo(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_categorical('n_steps', [128, 256, 512, 1024])
    gamma = trial.suggest_float('gamma', 0.9, 0.9999)
    gae_lambda = trial.suggest_float('gae_lambda', 0.8, 1.0)
    clip_range = trial.suggest_float('clip_range', 0.1, 0.3)
    ent_coef = trial.suggest_float('ent_coef', 1e-8, 1e-2, log=True)

    # Vectorized environment
    env = make_vec_env(f'CustomHopper-{mod_train}-v0', n_envs=4)

    # Create the PPO model
    model = PPO("MlpPolicy", env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_range=clip_range,
                ent_coef=ent_coef,
                verbose=1)

    # Train the model
    model.learn(total_timesteps=300_000)

    # Evaluate the model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

    env.close()

    return mean_reward


study = optuna.create_study(direction="maximize")
study.optimize(optimize_ppo, n_trials=30)

print("Best trial:")
print(study.best_trial)

best_params = study.best_trial.params

os.makedirs(args.name, exist_ok=True)

f = open(f'{args.name}/best_params.txt', 'w')
print(best_params, file=f)
f.close()

#env = make_vec_env(f'CustomHopper-{mod_train}-v0', n_envs=4)
#model = PPO("MlpPolicy", env, **best_params)
#model.learn(total_timesteps=100_000)

