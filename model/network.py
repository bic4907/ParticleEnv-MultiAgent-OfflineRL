import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils.model import fanin_init

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, activation=F.relu,
                 constrain_out=False):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        self.norm1 = nn.BatchNorm1d(input_dim)
        self.norm1.weight.data.fill_(1)
        self.norm1.bias.data.fill_(0)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        self.activation = activation

        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        if constrain_out:
            self.fc3.weight.data.uniform_(-0.003, 0.003)
            self.out_fn = torch.tanh
        else:
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        X = self.norm1(X)
        h1 = self.activation(self.fc1(X))
        h2 = self.activation(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out
