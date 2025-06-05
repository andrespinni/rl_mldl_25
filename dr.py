import train_CPD
import env.custom_hopper
import numpy as np
from env.mujoco_env import MujocoEnv

class DR(MujocoEnv):
    def __init__(self, N, envs):
        self.minMass = 0.7
        self.maxMass = 1.3
        self.minFriction = 0.8
        self.maxFriction = 1.2
        self.N = N
        self.envs = envs

    def set_random_parameters(self):
        """Set random masses"""
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """Sample masses, friction, and damping according to a domain randomization distribution"""
    
        params_list = []

        # Randomize masses (excluding root)
        mass_edges = np.linspace(self.minMass, self.maxMass, self.N + 1)

        # Randomize friction coefficients (for each geom, 3 values: sliding, torsional, rolling)
        friction_edges = np.linspace(self.minFriction, self.maxFriction, self.N + 1)

        for i in range(self.N):
            mass_range = (mass_edges[i], mass_edges[i+1])
            friction_range = (friction_edges[i], friction_edges[i+1])

            # Sample a value within the sub-domain for each parameter
            mass = np.random.uniform(*mass_range)
            friction = np.random.uniform(*friction_range)

            params_list.append({
                "mass": mass,
                "friction": friction,
            })

        return params_list
        
    def set_parameters(self, params):
        """Set each hopper link's mass, friction, and damping to new values"""
        
        for i, env in enumerate(self.envs):
            env.sim.model.body_mass[2:] = env.sim.model.body_mass[2:] * params[i]["mass"]
            self.sim.model.geom_friction[:] = self.sim.model.geom_friction[:] * params[i]["friction"]
