# Copyright Joe Worsham 2021

import numpy as np
import random

from collections import deque


class ExperienceReplayBuffer:
    """Experience Replay Buffer to learn on randomly
    sampled experiences. This helps training be breaking
    sequential correlations found in the episode trajectories.
    """
    def __init__(self, maxlen: int) -> None:
        """Create a new ExperienceReplayBuffer

        Args:
            maxlen (int): the maximum number of experiences to store
        """
        self._buffer = deque(maxlen=maxlen)

    def append(self, state, action, reward, next_state, done):
        """Append a single experience to the buffer as a tuple.

        Args:
            state (int):
            action (int):
            reward (float):
            next_state (int):
            done (bool):
        """
        self._buffer.append((state, action, reward, next_state, done))

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
        experiences = random.sample(self._buffer, batch_size)
        states = np.vstack([e[0] for e in experiences]).astype(np.int32)
        actions = np.vstack([e[1] for e in experiences]).astype(np.int32)
        rewards = np.vstack([e[2] for e in experiences]).astype(np.float32)
        next_states = np.vstack([e[3] for e in experiences]).astype(np.int32)
        dones = np.vstack([e[4] for e in experiences]).astype(np.uint8)
        return states, actions, rewards, next_states, dones

    def clear(self):
        """Clear the buffers
        """
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)

    def __str__(self) -> str:
        s = "Experience Replay Buffer:\nCount:\tState\tAction\tReward\tNext\tDone"
        for i, e in enumerate(self._buffer):
            s += f"\n{i}\t|{len(e[0])}|\t{e[1]}\t{e[2]}\t|{len(e[3])}|\t{e[4]}"
        return s
