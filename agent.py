import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64
        self.tanh = torch.nn.Tanh()

        """
            Actor network
        """
        self.fc1_actor = torch.nn.Linear(state_space, self.hidden)
        self.fc2_actor = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_actor_mean = torch.nn.Linear(self.hidden, action_space)
        self.fc3_actor_logstd = torch.nn.Linear(self.hidden, action_space)  # bau bau

        # Learned standard deviation for exploration at training time 
        # self.sigma_activation = F.softplus
        # init_sigma = 0.5
        # self.sigma = torch.nn.Parameter(torch.zeros(self.action_space)+init_sigma)


        """
            Critic network
        """
        # TASK 3: critic network for actor-critic algorithm
        self.fc1_critic = torch.nn.Linear(state_space, self.hidden)
        self.fc2_critic = torch.nn.Linear(self.hidden, self.hidden)
        self.fc3_critic_value = torch.nn.Linear(self.hidden, 1)

        # La dimensione finale dell'ultimo layer è 1 perché il critic network stima il valore scalare della funzione di valore dello stato V_w(s)

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)


    def forward(self, x):
        """
            Actor
        """
        x_actor = self.tanh(self.fc1_actor(x))
        x_actor = self.tanh(self.fc2_actor(x_actor))
        action_mean = self.fc3_actor_mean(x_actor)

        log_std = self.fc3_actor_logstd(x_actor) #bau bau
        sigma = F.softplus(log_std) #bau bau

        # sigma = self.sigma_activation(self.sigma)
        normal_dist = Normal(action_mean, sigma)


        """
            Critic
        """
        # TASK 3: forward in the critic network

        x_critic = self.tanh(self.fc1_critic(x))
        x_critic = self.tanh(self.fc2_critic(x_critic))
        state_value = self.fc3_critic_value(x_critic)

        return normal_dist, state_value


class Agent(object):
    def __init__(self, policy, device='cpu'):
        self.train_device = device
        self.policy = policy.to(self.train_device)
        #self.optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)  vecchio singolo optimizer
        #nuova verione: doppio opt:
        self.actor_params = list(self.policy.fc1_actor.parameters()) + \
                       list(self.policy.fc2_actor.parameters()) + \
                       list(self.policy.fc3_actor_mean.parameters()) + \
                    list(self.policy.fc3_actor_logstd.parameters())
         #    [self.policy.sigma]

        self.critic_params = list(self.policy.fc1_critic.parameters()) + \
                        list(self.policy.fc2_critic.parameters()) + \
                        list(self.policy.fc3_critic_value.parameters())
        
        #due optimizer con due lr diversi perché abbiamo visto che il critic "imoara" molto piu velocemente
        # dal grafico delle loss, poi puo essere un'idea osservare i gradienti
        self.actor_optimizer = torch.optim.Adam(self.actor_params, lr=1e-4)    # MODIFICA
        self.critic_optimizer = torch.optim.Adam(self.critic_params, lr=5e-4)

        self.gamma = 0.99
        # self.states = []
        # self.next_states = []
        # self.action_log_probs = []
        # self.rewards = []
        # self.done = []


    def update_policy(self, state, next_state, action_log_prob, reward, done):
        #action_log_probs = torch.stack(self.action_log_probs, dim=0).to(self.train_device).squeeze(-1)
        # states = torch.stack(self.states, dim=0).to(self.train_device).squeeze(-1)
        # next_states = torch.stack(self.next_states, dim=0).to(self.train_device).squeeze(-1)
        # rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        # done = torch.Tensor(self.done).to(self.train_device)

        state = torch.from_numpy(state).float().to(self.train_device)
        next_state = torch.from_numpy(next_state).float().to(self.train_device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.train_device)
        done = torch.tensor(done, dtype=torch.float32).to(self.train_device)
        #self.states, self.next_states, self.action_log_probs, self.rewards, self.done = [], [], [], [], []     

        #
        # TASK 3:
        #   - compute boostrapped discounted return estimates
        #   - compute advantage terms
        #   - compute actor loss and critic loss
        #   - compute gradients and step the optimizer
        #

        # Compute state values and next state values
        _, state_value = self.policy(state)
        _, next_state_value = self.policy(next_state)

        # Compute TD error (δt)
        td_targets = reward + self.gamma * next_state_value.squeeze(-1) * (1 - done)
        td_error = td_targets - state_value.squeeze(-1)
        # Stato t (state_values):
        # State value atteso per il futuro

        #stato t + 1 (td_targets):
        # State value atteso per il futuro (partendo da t+1) + reward effettiva dello stato precedente

        # Critic loss (Mean Squared Error)
        critic_loss = td_error.pow(2)

        # Actor loss (Policy Gradient with Advantage)
        advantage = td_error.detach()  # Detach to avoid backprop through critic
        #advantage = (advantages - advantages.mean()) / (advantages.std() + 1e-8) #normalizzati
        
        actor_loss = -(action_log_prob * advantage)

        # 1. Il vantaggio (o TD Error) viene calcolato usando il critic, poiché dipende da V(s_t) e V(s_{t+1}).
        # 2. Senza detach(), i gradienti della perdita dell'actor si propagherebbero attraverso il critic,
        #    modificandone i parametri in modo non desiderato.
        # 3. Questo creerebbe conflitti tra actor e critic, poiché hanno obiettivi diversi:
        #    - L'actor aggiorna la politica per massimizzare il vantaggio.
        #    - Il critic aggiorna i parametri per ridurre l'errore di stima del valore.
        # 4. Usando detach(), stacchiamo il TD Error dal grafo computazionale del critic.
        #    - Questo blocca il flusso dei gradienti attraverso il critic.
        #    - I parametri del critic rimangono invariati durante l'ottimizzazione dell'actor.
        # 5. In questo modo, l'actor e il critic vengono aggiornati in modo indipendente,
        #    preservando la stabilità e l'efficacia dell'apprendimento.

        # Total loss
        total_loss = actor_loss + critic_loss

        self.actor_loss = actor_loss.item()  # Converte il tensore in un float
        self.critic_loss = critic_loss.item()
        self.total_loss = total_loss.item()
        
        # Optimize both actor and critic
        #vecchia versione singolo opt
        #self.optimizer.zero_grad()
        #total_loss.backward()
        #self.optimizer.step()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_params, max_norm=0.5)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_params, max_norm=0.5)
        self.critic_optimizer.step()

        return 


    def get_action(self, state, evaluation=False):
        """ state -> action (3-d), action_log_densities """
        x = torch.from_numpy(state).float().to(self.train_device)

        normal_dist, _ = self.policy(x)

        if evaluation:  # Return mean
            return normal_dist.mean, None

        else:   # Sample from the distribution
            action = normal_dist.sample()

            # Compute Log probability of the action [ log(p(a[0] AND a[1] AND a[2])) = log(p(a[0])*p(a[1])*p(a[2])) = log(p(a[0])) + log(p(a[1])) + log(p(a[2])) ]
            action_log_prob = normal_dist.log_prob(action).sum()

            return action, action_log_prob


    def store_outcome(self, state, next_state, action_log_prob, reward, done):
        self.states.append(torch.from_numpy(state).float())
        self.next_states.append(torch.from_numpy(next_state).float())
        self.action_log_probs.append(action_log_prob)
        self.rewards.append(torch.Tensor([reward]))
        self.done.append(done)

        #VAI

