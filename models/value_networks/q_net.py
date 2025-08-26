import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../../utils")
import utils.nn_utils as nn_utils


class Q(nn.Module):
    """
    Class Q implements an action-value network using an MLP function
    approximator. The action value is computed by concatenating the action to
    the state observation as the input to the neural network.
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
        super(Q, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(lambda module: nn_utils.weights_init_(module, init))

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        else:
            raise ValueError(f"unknown activation {activation}")

    def forward(self, state, action):
        """
        Performs the forward pass through each network, predicting the
        action-value for `action` in `state`.

        Parameters
        ----------
        state : torch.Tensor of float
            The state that the action was taken in
        action : torch.Tensor of float
            The action taken in the input state to predict the value function
            of

        Returns
        -------
        torch.Tensor
            The action value prediction
        """
        xu = torch.cat([state, action], 1)

        x = self.act(self.linear1(xu))
        x = self.act(self.linear2(x))
        x = self.linear3(x)

        return x


