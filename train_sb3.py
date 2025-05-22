"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import os
import gym
import argparse
from env.custom_hopper import *
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', default=1_000_000, type=int, help='Number of training episodes')
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')
    parser.add_argument('--name', default='hopper-train_noName', type=str, help='Scegliere nome')
    parser.add_argument("--mod_train", default="source",type=str)
    parser.add_argument("--mod_eval", default="source",type=str)

    
    return parser.parse_args()

args = parse_args()

mod_train=args.mod_train
mod_eval=args.mod_eval

def main():
    # Crea un nuovo run W&B per tenere traccia del training.
    #config permette di centralizzare i parametri che poi usi nel codice.
    
    wandb.init(
        project="PPO",
        name=f"{args.name}_{mod_train}",
        entity="andrea-gaudino02-politecnico-di-torino",
        config={
            "env": "CustomHopper-source-v0",
            "algorithm": "PPO",
            "total_timesteps": args.episodes,
            "eval_freq": 10_000,
            "n_eval_episodes": 5
        },
        sync_tensorboard=True,
        monitor_gym=True
    )
    config = wandb.config
    

    
    # Create training and evaluation environments
    # Uno per il training, uno per la valutazione periodica
    train_env = gym.make(f'CustomHopper-{mod_train}-v0')
    eval_env = gym.make(f'CustomHopper-{mod_eval}-v0')

    # Utile per capire le dimensioni dello stato e dell’azione.
    #get_parameters() è una funzione custom che stampa, ad esempio, le masse dei link
    

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper
    
    
    #log dei dettagli
    os.makedirs(args.name, exist_ok=True)
    out_file_name = f"{args.name}/dettagli_train-sb3.txt"
    out_file = open(out_file_name, "w")
    
    
    out_file.write(f"Action space: {train_env.action_space}\n")
    out_file.write(f"State space: {train_env.observation_space}\n")
    out_file.write(f"Dynamic parameters: {train_env.get_parameters()}\n")
    
    out_file.write(f"\nModel trainato con {mod_train}")
    out_file.write(f"Model eval con {mod_eval}\n")
        
    

    #
    # TASK 4 & 5: train and test policies on the Hopper env with stable-baselines3
    #

    # Initialize the PPO model
    # MlpPolicy: rete neurale multilayer perceptron.
    #tensorboard_log: salva i log compatibili con TensorBoard
    
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=f"{args.name}/runs/"
    )

    # Define evaluation callback: test the agent every eval_freq steps
    # Salva il miglior modello ogni eval_freq passi.
    # Fa la valutazione su n_eval_episodes episodi

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{args.name}/best_model/",
        log_path=f"{args.name}/eval_logs/",
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True, # utile nell'eval, mentre nel train è stocastico
        render=args.render
        )

    # Define Weights & Biases callback
    # Salva i modelli e logga automaticamente i grafici su W&B
    wandb_callback = WandbCallback(
        model_save_path=f"{args.name}/models/",
        verbose=2
    )

    # Combine callbacks
    # Combina entrambi i callback in una sola lista da passare alla funzione .learn()
    callback = CallbackList([wandb_callback, eval_callback])

    # Train the model
    # Addestra PPO per il numero di passi specificato nel config.
    #I callback vengono attivati durante il processo
    
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callback
    )

    # Final evaluation
    #Dopo il training, valuta il modello e stampa la media e deviazione standard della ricompensa
    
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True
    )
    print(f"Final evaluation mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    
    out_file.write(f"Final evaluation mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    out_file.close()

    # Finish the W&B run
    wandb.finish()

if __name__ == '__main__':
    main()