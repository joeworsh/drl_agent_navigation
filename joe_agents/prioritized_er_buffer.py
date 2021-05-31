# Copyright Joe Worsham 2021

import numpy as np
import random

from collections import deque


class PrioritizedExperienceReplayBuffer:
    """Experience Replay Buffer to learn on randomly
    sampled experiences. This helps training be breaking
    sequential correlations found in the episode trajectories.
    """
    def __init__(self, maxlen: int, value_function, gamma: float, a_damp: float, e: float=1e-1) -> None:
        """Create a new ExperienceReplayBuffer

        Args:
            maxlen (int): the maximum number of experiences to store
            value_function: callable to get action values for state
            gamma (float): the discount rate to use when computing priority
            a_damp (float): damping effect for the priorities. Between zero and one.
            e (float): small constant to resist sample starvation. Defaults to 1e-1.
        """
        self._buffer = deque(maxlen=maxlen)
        self._value_function = value_function
        self._gamma = gamma
        self._a_damp = a_damp
        self._e = e

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, gamma):
        self._gamma = gamma

    @property
    def a_damp(self):
        return self._a_damp

    @a_damp.setter
    def a_damp(self, a_damp):
        self._a_damp = a_damp

    @property
    def e(self):
        return self._e

    @e.setter
    def a_damp(self, e):
        self._e = e

    def append(self, state, action, reward, next_state, done):
        """Append a single experience to the buffer as a tuple.

        Args:
            state (int):
            action (int):
            reward (float):
            next_state (int):
            done (bool):
        """
        dt = reward + self.gamma * np.amax(self._value_function(next_state)) - self._value_function(state)[action]
        dt = np.abs(dt)
        self._buffer.append((dt, state, action, reward, next_state, done))

    def extend(self, trajectory):
        """Add the experiences found in this
        trajectory to the buffer. This will
        disassociate each experience with the
        others in the trajectory.

        Args:
            trajectory (list): a list of experiences to add
        """
        self._buffer.extend(trajectory)

    def sample(self, batch_size: int):
        """Sample a random, uncorrelated batch
        of experiences from the buffer. Samples
        without replacement.

        Args:
            batch_size (int): The number of samples to draw

        Returns:
            list: A collection of uncorrelated experiences
        """
        # prioritize the buffer
        priorities = [(e[0] + self.e)**self.a_damp for e in self._buffer]
        sum_p = np.sum(priorities)
        priorities = [p / sum_p for p in priorities]

        experiences = random.choices(self._buffer, weights=priorities, k=batch_size)
        priorities = np.vstack([e[0] for e in experiences]).astype(np.float32)
        states = np.vstack([e[1] for e in experiences]).astype(np.float32)
        actions = np.vstack([e[2] for e in experiences]).astype(np.int32)
        rewards = np.vstack([e[3] for e in experiences]).astype(np.float32)
        next_states = np.vstack([e[4] for e in experiences]).astype(np.float32)
        dones = np.vstack([e[5] for e in experiences]).astype(np.uint8)
        return priorities, states, actions, rewards, next_states, dones

    def clear(self):
        """Clear the buffers
        """
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)

    def __str__(self) -> str:
        s = "Experience Replay Buffer:\nCount:\tState\tAction\tReward\tNext\tDone"
        for i, e in enumerate(self._buffer):
            s += f"\n{i})\t|{len(e[0])}|\t{e[1]}\t{e[2]}\t|{len(e[3])}|\t{e[4]}"
        return s
