"""Sample script for training a control policy on the Hopper environment
   using stable-baselines3 (https://stable-baselines3.readthedocs.io/en/master/)

    Read the stable-baselines3 documentation and implement a training
    pipeline with an RL algorithm of your choice between PPO and SAC.
"""
import gym
from env.custom_hopper import *
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy

def main():
    # Crea un nuovo run W&B per tenere traccia del training.
    #config permette di centralizzare i parametri che poi usi nel codice.
    
    wandb.init(
        project="hopper_ppo",
        config={
            "env": "CustomHopper-source-v0",
            "algorithm": "PPO",
            "total_timesteps": 100_000,
            "eval_freq": 1_000,
            "n_eval_episodes": 5
        }
    )
    config = wandb.config

    # Create training and evaluation environments
    # Uno per il training, uno per la valutazione periodica
    train_env = gym.make('CustomHopper-source-v0')
    eval_env = gym.make('CustomHopper-source-v0')

    # Utile per capire le dimensioni dello stato e dell’azione.
    #get_parameters() è una funzione custom che stampa, ad esempio, le masse dei link
    

    print('State space:', train_env.observation_space)  # state-space
    print('Action space:', train_env.action_space)  # action-space
    print('Dynamics parameters:', train_env.get_parameters())  # masses of each link of the Hopper

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
        tensorboard_log="runs/"
    )

    # Define evaluation callback: test the agent every eval_freq steps
    # Salva il miglior modello ogni eval_freq passi.
    # Fa la valutazione su n_eval_episodes episodi

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="best_model/",
        log_path="eval_logs/",
        eval_freq=config.eval_freq,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=True, # utile nell'eval, mentre nel train è stocastico
        render=False
    )

    # Define Weights & Biases callback
    # Salva i modelli e logga automaticamente i grafici su W&B
    wandb_callback = WandbCallback(
        model_save_path="models/",
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

    # Finish the W&B run
    wandb.finish()

if __name__ == '__main__':
    main()