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
        return np.stack(random.sample(self._buffer, batch_size))
        

    def clear(self):
        """Clear the buffers
        """
        self._buffer.clear()

    def __len__(self):
        return len(self._buffer)
