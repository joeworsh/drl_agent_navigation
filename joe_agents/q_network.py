# Copyright Joe Worsham 2021

from torch import nn


class QNetwork(nn.Module):
    """Basic Deep Q-Network implementation for training.
    This class itself is not too fancy - it simply models
    all the action values for each possible action in a
    given state.
    """

    def __init__(self, state_size, action_size, hidden_nodes=None):
        """Create a new Q-Network with the configured sizes.
        If layers is not provided it defaults to [521, 256]

        Args:
            state_size (int): The number of features in the state space
            action_size (int): The number of discrete action values to model
            hidden_nodes (list, optional): List of hidden node counts per layer. Defaults to None.
        """
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_nodes = hidden_nodes if hidden_nodes is not None else [
            521, 256]

        self.relu = nn.ReLU()
        self.in_layer = nn.Linear(self.state_size, self.hidden_nodes[0])
        hidden_layers = []
        for i in range(len(self.hidden_nodes) - 1):
            hidden_layers.append(nn.Linear(self.hidden_nodes[i], self.hidden_nodes[i+1]))
            hidden_layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*hidden_layers)
        self.out_layer = nn.Linear(self.hidden_nodes[-1], self.action_size)

    def forward(self, x):
        """Forward implementation to compute action values
        for each discrete action based on the current state,
        x.

        Args:
            x (array): The current state of the environment.
        """
        x = self.in_layer(x)
        x = self.relu(x)
        x = self.hidden(x)
        return self.out_layer(x)
