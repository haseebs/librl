import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../../utils")
import utils.nn_utils as nn_utils


class V(nn.Module):
    """
    Class V is an MLP for estimating the state value function `v`.
    """
    def __init__(self, num_inputs, hidden_dim, init, activation):
        """
        Constructor

        Parameters
        ----------
        num_inputs : int
            Dimensionality of input feature vector
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
        super(V, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(lambda module: nn_utils.weights_init_(module, init))

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation {activation}")

    def forward(self, state):
        """
        Performs the forward pass through the network, predicting the value of
        `state`.

        Parameters
        ----------
        state : torch.Tensor of float
            The feature vector of the state to compute the value of

        Returns
        -------
        torch.Tensor of float
            The value of the state
        """
        x = self.act(self.linear1(state))
        x = self.act(self.linear2(x))
        x = self.linear3(x)
        return x


