"""Implementation of the Hopper environment supporting
domain randomization optimization.
    
    See more at: https://www.gymlibrary.dev/environments/mujoco/hopper/
"""
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv


class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)

        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses
        self.original_friction = np.copy(self.sim.model.geom_friction)  # Default friction (n_geom, 3)
        self.original_damping = np.copy(self.sim.model.dof_damping)     # Default damping (n_dof,)

        if domain == 'source':  # Source environment has an imprecise torso mass (-30% shift)
            self.sim.model.body_mass[1] *= 0.7
    '''    
    def set_random_parameters(self):
        """Set random masses"""
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """Sample masses, friction, and damping according to a domain randomization distribution"""

        # Randomize masses (excluding root)
        masses = np.copy(self.original_masses)
        mass_scale = np.random.uniform(low=0.7, high=1.3, size=len(masses))
        masses = masses * mass_scale

        # Randomize friction coefficients (for each geom, 3 values: sliding, torsional, rolling)
        friction = np.copy(self.original_friction)
        friction_scale = np.random.uniform(low=0.6, high=1.2, size=friction.shape)
        friction = friction * friction_scale

        # Randomize damping (for each degree of freedom)
        damping = np.copy(self.original_damping)
        damping_scale = np.random.uniform(low=0, high=1, size=damping.shape)
        damping = damping * damping_scale

        # Return as a dictionary for clarity
        return {
            "masses": masses,
            "friction": friction,
            "damping": damping
        }

    def set_parameters(self, params):
        """Set each hopper link's mass, friction, and damping to new values"""
        self.sim.model.body_mass[1:] = params["masses"]
        self.sim.model.geom_friction[:] = params["friction"]
        self.sim.model.dof_damping[:] = params["damping"]
    '''

    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}


    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    
    def reset_model(self):
        """Reset the environment to a random initial state"""
        self.set_random_parameters()
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()


    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20


    def set_mujoco_state(self, state):
        """Set the simulator to a specific state

        Parameters:
        ----------
        state: ndarray,
               desired state
        """
        mjstate = deepcopy(self.get_mujoco_state())

        mjstate.qpos[0] = 0.
        mjstate.qpos[1:] = state[:5]
        mjstate.qvel[:] = state[5:]

        self.set_sim_state(mjstate)


    def set_sim_state(self, mjstate):
        """Set internal mujoco state"""
        return self.sim.set_state(mjstate)


    def get_mujoco_state(self):
        """Returns current mjstate"""
        return self.sim.get_state()



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

