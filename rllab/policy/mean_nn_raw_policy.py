from rllab.policy.base import DeterministicPolicy
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from rllab.misc import autoargs
from rllab.misc.ext import compile_function
import itertools
import theano
import numpy as np


class MeanNNRawPolicy(DeterministicPolicy, Serializable):
    @autoargs.arg('hidden_sizes', type=int, nargs='*',
                  help='list of sizes for the fully-connected hidden layers')
    @autoargs.arg('hidden_nl', type=str,
                  help='list of nonlinearities for the hidden layers')
    @autoargs.arg('hidden_init', type=str,
                  help='list of initializers for the hidden layers weights')
    @autoargs.arg('output_nl', type=str,
                  help='nonlinearity for the output layer')
    @autoargs.arg('output_init', type=str,
                  help='initializer for the output layer weights')
    @autoargs.arg('bn', type=bool,
                  help='whether to apply batch normalization to all layers')
    # pylint: disable=dangerous-default-value
    def __init__(
            self,
            mdp,
            hidden_sizes=[100, 100],
            hidden_nl='rectify',
            hidden_init='he_uniform',
            output_nl='none',
            output_init='he_uniform',
            bn=False):
        fan_in_sizes = [mdp.observation_shape[0]] + hidden_sizes[:-1]
        params = []
        layers = []
        for idx, hidden_size, fan_in_size in zip(
                itertools.count(), hidden_sizes, fan_in_sizes):
            if hidden_init == 'he_uniform':
                bound = 1.0 / np.sqrt(fan_in_size)
                W_init = np.random.uniform(
                    -bound, bound, size=(fan_in_size, hidden_size))
                b_init = np.random.uniform(
                    -bound, bound, size=(hidden_size,))
            else:
                raise NotImplementedError
            W = theano.shared(W_init, name="hidden_%d_W" % idx)
            b = theano.shared(b_init, name="hidden_%d_b" % idx)
            params.append(W)
            params.append(b)
            layers.append((W, b))
        if output_init == 'he_uniform':
            bound = 1.0 / np.sqrt(fan_in_size)
            W_init = np.random.uniform(
                -bound, bound, size=(fan_in_size, hidden_size))
            b_init = np.random.uniform(
                -bound, bound, size=(hidden_size,))
        self._hidden_nl = hidden_nl
        self._output_nl = output_nl
