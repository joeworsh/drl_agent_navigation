# Copyright Joe Worsham 2021

import numpy as np
import random


class EGreedyPolicy:
    """A policy for implementing an epsilon-greedy strategy.
    With epsilon likelihood a random action is drawn. With
    a 1-epsilon likelihood an estimated optimal action is
    drawn.
    """
    def __init__(self, action_size: int, epsilon: float, max_policy) -> None:
        """Create a new EGreedyPolicy.

        Args:
            action_size (int): the number of actions to choose from
            epsilon (float): the starting epsilon value
            max_policy (callable): The estimated optimal policy
        """
        self._action_size = action_size
        self._epsilon = epsilon
        self._max_policy = max_policy

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, epsilon):
        self._epsilon = epsilon

    def __call__(self, state):
        """Return a random or optimal action.

        Args:
            state (np.Array): the current state to act upon

        Returns:
            int: the selected discrete action
        """
        if random.random() > self._epsilon:
            return self._max_policy(state)
        return random.choice(np.arange(self._action_size))
