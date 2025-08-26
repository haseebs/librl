import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../../utils")
import utils.nn_utils as nn_utils


class DiscreteQ(nn.Module):
    """
    Class DiscreteQ implements an action value network with number of
    predicted action values equal to the number of available actions.
    """
    def __init__(self, num_inputs, num_actions, hidden_dim, init,
                 activation):
        """
        Constructor

        Parameters
        ----------
        num_inputs : int
            Dimensionality of state feature vector
        num_actions : int
            Dimensionality of the action feature vector
        hidden_dim : int
            The number of units in each hidden layer
        init : str
            The initialization scheme to use for the weights, one of
            'xavier_uniform', 'xavier_normal', 'uniform', 'normal',
            'orthogonal', by default None. If None, leaves the default
            PyTorch initialization.
        activation : str
            The activation function to use; one of 'relu', 'tanh'
        """
        super(DiscreteQ, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        self.apply(lambda module: nn_utils.weights_init_(module, init))

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation {activation}")

    def forward(self, state):
        """
        Performs the forward pass through each network, predicting the
        action-value for `action` in `state`.

        Parameters
        ----------
        state : torch.Tensor of float
            The state that the action was taken in

        Returns
        -------
        torch.Tensor
            The action value predictions
        """

        x = self.act(self.linear1(state))
        x = self.act(self.linear2(x))
        return self.linear3(x)


