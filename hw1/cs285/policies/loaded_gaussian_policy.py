import numpy as np

from cs285.infrastructure import pytorch_util as ptu
from .base_policy import BasePolicy
from torch import nn
import torch
import pickle


def create_linear_layer(W, b) -> nn.Linear:
    """
    Takes
        - W: (out_features x in_features) numpy array
        - b: (1 x out_features) numpy array
    Returns
        - linear_layer: nn.Linear, PyTorch dense layer
    """
    # in_features, out_features = W.shape
    out_features, in_features = W.shape
    linear_layer = nn.Linear(
        in_features,
        out_features,
    )
    linear_layer.weight.data = ptu.from_numpy(W.T)
    linear_layer.bias.data = ptu.from_numpy(b[0])
    return linear_layer


def read_layer(l):
    """
    Another helper function for this module only.
    No idea what it does. Maybe bookkeeping.
    """
    assert list(l.keys()) == ['AffineLayer']
    assert sorted(l['AffineLayer'].keys()) == ['W', 'b']
    return l['AffineLayer']['W'].astype(np.float32), l['AffineLayer'][
        'b'].astype(np.float32)


class LoadedGaussianPolicy(BasePolicy, nn.Module):
    def __init__(self, filename, **kwargs):
        super().__init__(**kwargs)

        # Open the pickled expert policy.
        with open(filename, 'rb') as f:
            data = pickle.loads(f.read())

        # Define the hidden layer activation functions.
        self.nonlin_type = data['nonlin_type']
        if self.nonlin_type == 'lrelu':
            self.non_lin = nn.LeakyReLU(0.01)
        elif self.nonlin_type == 'tanh':
            self.non_lin = nn.Tanh()
        else:
            raise NotImplementedError()

        # Assert that this loaded policy is a "GaussianPolicy"
        policy_type = [k for k in data.keys() if k != 'nonlin_type'][0]
        assert policy_type == 'GaussianPolicy', (
            'Policy type {} not supported'.format(policy_type)
        )
        self.policy_params = data[policy_type]

        # The loaded policy has policy_params
        # policy_params is a dictionary with these 4 entries.
        assert set(self.policy_params.keys()) == {
            'logstdevs_1_Da', 'hidden', 'obsnorm', 'out'
        }

        # Build the policy. First, observation normalization.
        # Under the loaded policy, the observations are (approx) distributed as
        # N(obsnorm_mean, obsnorm_stdev)
        assert list(self.policy_params['obsnorm'].keys()) == ['Standardizer']
        obsnorm_mean = self.policy_params['obsnorm']['Standardizer']['mean_1_D']
        obsnorm_meansq = self.policy_params['obsnorm']['Standardizer'][
            'meansq_1_D']
        obsnorm_stdev = np.sqrt(
            np.maximum(0, obsnorm_meansq - np.square(obsnorm_mean)))
        print('obs', obsnorm_mean.shape, obsnorm_stdev.shape)

        self.obs_norm_mean = nn.Parameter(ptu.from_numpy(obsnorm_mean))
        self.obs_norm_std = nn.Parameter(ptu.from_numpy(obsnorm_stdev))

        # Reconstruct the hidden layers froam the loaded data.
        self.hidden_layers = nn.ModuleList()

        # The 'hidden' layers must be "FeedforwardNet" type
        # The layers are kept in `layer_params` dictionary, ordered by the keys.
        # They are read out, made into PyTorch layers, then appended, in order.
        assert list(self.policy_params['hidden'].keys()) == ['FeedforwardNet']
        layer_params = self.policy_params['hidden']['FeedforwardNet']
        for layer_name in sorted(layer_params.keys()):
            l = layer_params[layer_name]
            W, b = read_layer(l)
            linear_layer = create_linear_layer(W, b)
            self.hidden_layers.append(linear_layer)

        # Output layer (does not have an activation function).
        W, b = read_layer(self.policy_params['out'])
        self.output_layer = create_linear_layer(W, b)

    def forward(self, obs):
        normed_obs = (obs - self.obs_norm_mean) / (self.obs_norm_std + 1e-6)
        h = normed_obs
        for layer in self.hidden_layers:
            h = layer(h)
            h = self.non_lin(h)
        return self.output_layer(h)

    ##################################

    def update(self, obs_no, acs_na, adv_n=None, acs_labels_na=None):
        raise NotImplementedError("""
            This policy class simply loads in a particular type of policy and
            queries it. Do not try to train it.
        """)

    def get_action(self, obs):
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None, :]
        observation = torch.from_numpy(observation.astype(np.float32))
        action = self(observation)
        # For  classes that inherit from nn.Module,
        # self(...) implicitly calls self.forward(...)
        return action.detach().numpy()

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
