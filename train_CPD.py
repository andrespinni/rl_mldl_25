import gym
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, TensorDataset
from env.custom_hopper import *


SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    
class MLPWrapper:
    """Wrap a PyTorch MLP policy for Gym-style evaluation"""
    def __init__(self, mlp, device='cpu'):
        self.mlp = mlp
        self.device = device

    def __call__(self, obs):
        t = torch.tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            out = self.mlp(t.unsqueeze(0))
        return out.cpu().numpy()[0]

class CyclicPolicyDistillationPPO:
    def __init__(
        self,
        env_id,
        N=3,
        local_steps=100000,
        rollout_per_domain=2048,
        distill_epochs=10,
        distill_batch=64,
        lr=3e-4,
        device='cpu',
        early_stop_delta=1.0,
        z_threshold=1.96,
        early_stop_patience=3
    ):
        
        self.N = N
        self.local_steps = local_steps
        self.rollout_per_domain = rollout_per_domain
        self.distill_epochs = distill_epochs
        self.distill_batch = distill_batch
        self.device = device
        self.early_stop_delta = early_stop_delta
        self.z_threshold = z_threshold
        self.early_stop_patience = early_stop_patience

        # sub-domain PPO agents
        self.envs = [gym.make(env_id) for _ in range(N)]
        self.models = [PPO('MlpPolicy', env, verbose=0) for env in self.envs]

        # global MLP policy for distillation
        obs_dim = self.envs[0].observation_space.shape[0]
        act_dim = self.envs[0].action_space.shape[0]
        self.global_mlp = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim)
        ).to(self.device)
        self.optimizer = optim.Adam(self.global_mlp.parameters(), lr=lr)

    def local_training_cycle(self):
        order = list(range(self.N)) + list(range(self.N-1, -1, -1))
        for idx in order:
            print(f"Training local PPO domain {idx}")
            print(f"Training local PPO domain {idx}",file=file)
            self.models[idx].learn(total_timesteps=self.local_steps)

    def collect_distill_data(self):
        obs_buf, act_buf = [], []
        for idx, model in enumerate(self.models):
            env = self.envs[idx]
            obs = env.reset()
            collected = 0
            while collected < self.rollout_per_domain:
                action, _ = model.predict(obs, deterministic=True)
                obs_buf.append(obs.copy())
                act_buf.append(action)
                obs, _, done, _ = env.step(action)
                collected += 1
                if done:
                    obs = env.reset()
        obs_t = torch.tensor(np.array(obs_buf), dtype=torch.float32).to(self.device)
        act_t = torch.tensor(np.array(act_buf), dtype=torch.float32).to(self.device)
        return DataLoader(TensorDataset(obs_t, act_t), batch_size=self.distill_batch, shuffle=True)

    def distill_global(self):
        """Train global_mlp to mimic local PPO actions (behavior cloning)"""
        loader = self.collect_distill_data()
        loss_fn = nn.MSELoss()
        for epoch in range(self.distill_epochs):
            total_loss = 0.0
            for obs_b, act_b in loader:
                preds = self.global_mlp(obs_b)
                loss = loss_fn(preds, act_b)
                self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
                total_loss += loss.item() * obs_b.size(0)
            avg_loss = total_loss / len(loader.dataset)
            print(f"Distill MLPolicy epoch {epoch+1}/{self.distill_epochs}, loss: {avg_loss:.6f}")
            print(f"Distill MLPolicy epoch {epoch+1}/{self.distill_epochs}, loss: {avg_loss:.6f}",file=file)


    def evaluate_policy(self, policy_fn, env_id, episodes=20):
        env = gym.make(env_id)
        rewards = []
        for _ in range(episodes):
            obs = env.reset()
            done = False
            total_r = 0.0
            while not done:
                action = policy_fn(obs)
                obs, r, done, _ = env.step(action)
                total_r += r
            rewards.append(total_r)
        return np.mean(rewards), np.std(rewards)

    def run(self, max_cycles=30, eval_episodes=20, fine_tune_steps=100000):
        target_env = "CustomHopper-target-v0"
        best_mean = -np.inf
        patience = 0

        for cycle in range(max_cycles):
            print(f"\n=== Cycle {cycle+1}/{max_cycles} ===")
            print(f"\n=== Cycle {cycle+1}/{max_cycles} ===",file=file)
            # 1) local training
            self.local_training_cycle()
            # 2) distill
            self.distill_global()
            # 3a) evaluate distilled MLP
            mlp_wrap = MLPWrapper(self.global_mlp, self.device)
            mean_mlp, std_mlp = self.evaluate_policy(mlp_wrap, target_env, eval_episodes)
            print(f"Distilled MLP zero-shot: mean {mean_mlp:.2f} ± {std_mlp:.2f}")
            print(f"Distilled MLP zero-shot: mean {mean_mlp:.2f} ± {std_mlp:.2f}",file=file)
            # early-stopping check on MLP performance
            if best_mean > -np.inf:
                z_score = (mean_mlp - best_mean) / (std_mlp + 1e-8)
                print(f"Improvement z-score: {z_score:.2f} (threshold {self.z_threshold})")
                print(f"Improvement z-score: {z_score:.2f} (threshold {self.z_threshold})",file=file)
                if z_score < self.z_threshold:
                    patience += 1
                    print(f"No significant improvement, patience {patience}/{self.early_stop_patience}")
                    print(f"No significant improvement, patience {patience}/{self.early_stop_patience}",file=file)
                else:
                    best_mean = mean_mlp
                    patience = 0
                    print("Significant improvement, resetting patience.")
                    print("Significant improvement, resetting patience.",file=file)
                if patience >= self.early_stop_patience:
                    print("Early stopping triggered.")
                    print("Early stopping triggered.",file=file)
                    break
            else:
                best_mean = mean_mlp
                print("Initial mean set.")
                print("Initial mean set.",file=file)

        # save distilled MLP
        torch.save(self.global_mlp.state_dict(), "OUTPUT_CPD/ppo_global_distilled.pth")

        """
        # 3b) fine-tuning with PPO using distilled weights
        print("\n=== Fine-tuning global PPO ===")
        env = gym.make(target_env)
        ppo_global = PPO('MlpPolicy', env, verbose=0)
        # load distilled weights into PPO
        ppo_global.policy.mlp.load_state_dict(torch.load("ppo_global_distilled.pth"))
        ppo_global.learn(total_timesteps=fine_tune_steps)
        ppo_global.save("ppo_global_finetuned")

        # evaluate fine-tuned PPO
        ppo_fn = lambda obs: ppo_global.predict(obs, deterministic=True)[0]
        mean_ppo, std_ppo = self.evaluate_policy(ppo_fn, target_env, eval_episodes)
        print(f"Fine-tuned PPO: mean {mean_ppo:.2f} ± {std_ppo:.2f}")
        """

if __name__ == '__main__':
    os.makedirs("OUTPUT_CPD",exist_ok=True)
    file = open("OUTPUT_CPD/log.txt","w")
    trainer = CyclicPolicyDistillationPPO(env_id="CustomHopper-v0")
    trainer.run()
    file.close()
