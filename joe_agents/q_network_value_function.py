# Copyright Joe Worsham 2021

import numpy as np
import torch

from joe_agents.q_network import QNetwork


class QNetworkValueFunction:
    """A value function based off of a PyTorch Q-Network.
    """
    def __init__(self, network: QNetwork, device: str) -> None:
        """Create a new QNetworkValueFunction that will
        return the estimated values of every action.

        Args:
            network (QNetwork): The Q-Network for decision making
            device (str): The hardware device the network is on
        """
        self._network = network
        self._device = device

    def __call__(self, state):
        """Get the array of all action values for the given state.

        Args:
            state (np.Array): the current state to act on

        Returns:
            [np.Array]: list of values for each action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(self._device)
        self._network.eval()
        with torch.no_grad():
            action_values = self._network(state)
        self._network.train()

        vals = action_values.cpu().data.numpy().squeeze()
        return vals
