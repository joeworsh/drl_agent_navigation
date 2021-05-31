# Copyright Joe Worsham 2021

import gym
import numpy as np

from gym.spaces import Box, Discrete
from unityagents import UnityEnvironment

# keys for the environment config
EXECUTABLE = "executable"
TRAIN_MODE = "train_mode"


class BananaEnv(gym.Env):
    """An OpenAI Gym for the Banana Unity environment.
    """
    def __init__(self, env_config):
        """Create a new BananaEnv pointing at the compiled
        Unity environment.

        Args:
            env_config (dict): Environment configuration
        """
        executable = env_config[EXECUTABLE]
        train_mode = env_config[TRAIN_MODE]
        self._env = UnityEnvironment(file_name=executable)
        self._brain_name = self._env.brain_names[0]
        self._brain = self._env.brains[self._brain_name]
        self._train_mode = train_mode
        high = np.tile(np.array([np.finfo(np.float32).max,]), [37,])
        self.observation_space = Box(-high, high, dtype=np.float32)
        self.action_space = Discrete(4)

    def step(self, action):
        """Transition from one state to the next with
        the given action.

        Args:
            action (int): the action to perform

        Returns:
            tuple: next_state, reward, done, additional_info
        """
        env_info = self._env.step(int(action))[self._brain_name]        
        next_state = env_info.vector_observations[0]   
        reward = env_info.rewards[0]                   
        done = env_info.local_done[0]   
        return next_state, reward, done, {}           

    def reset(self):
        """Reset the Unity environment.

        Returns:
            int: the starting state
        """
        env_info = self._env.reset(train_mode=self._train_mode)[self._brain_name]
        return env_info.vector_observations[0]

    def close(self):
        """Close the Unity environment and shut down
        the process.
        """
        self._env.close()