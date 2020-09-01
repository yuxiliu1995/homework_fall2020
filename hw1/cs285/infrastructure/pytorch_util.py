from typing import Union, Optional
import torch
import numpy as np
from torch import nn

Activation = Union[str, nn.Module]


def _identity(x):
    return x


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
}

def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Optional[Activation] = None,
):
    """
        Builds a feedforward neural network

        arguments:
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            model: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    layers = []

    if n_layers == 0: # no hidden layers
        layers += [nn.Linear(input_size, output_size)]
    else:
        layers += [nn.Linear(input_size, size)]
        layers += [activation, nn.Linear(size, size)] * (n_layers - 1)
        layers += [activation]
        layers += [nn.Linear(size, output_size)]

    if output_activation: # final layer activation
        layers += [output_activation]

    return torch.nn.Sequential(*layers)

def from_numpy(array):
    return torch.from_numpy(array.astype(np.float32))
