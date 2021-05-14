# Copyright Joe Worsham 2021

import numpy as np
import torch

from joe_agents.q_network import QNetwork


class QNetworkPolicy:
    """A policy based off of a PyTorch Q-Network.
    """
    def __init__(self, network: QNetwork, device: str) -> None:
        """Create a new QNetworkPolicy that will
        act optimally based on estimated action
        values.

        Args:
            network (QNetwork): The Q-Network for decisio nmaking
            device (str): The hardware device the network is on
        """
        self._network = network
        self._device = device

    def __call__(self, state):
        """Invoke to get the next action to perform.
        Will always return the most optimal action.

        Args:
            state (np.Array): the current state to act on

        Returns:
            [int]: the discrete action to perform
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._network.eval()
        with torch.no_grad():
            action_values = self._network(state)
        self._network.train()

        return np.argmax(action_values.cpu().data.numpy())
