import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne
import numpy as np
import theano
import theano.tensor as TT
from pydoc import locate
from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
# from rllab.core.lasagne_layers import batch_norm
from rllab.core.parameterized import Parameterized
from rllab.core.serializable import Serializable
from rllab.policy.base import StochasticPolicy, Policy
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import autoargs
from rllab.sampler import parallel_sampler


class UniformControlPolicy(Policy, Serializable):

    def __init__(
            self,
            mdp,
    ):
        Serializable.quick_init(self, locals())
        super(UniformControlPolicy, self).__init__(mdp)
        Parameterized.__init__(self)
        self.action_bounds = mdp.action_bounds

    def get_params_internal(self, **tags):
        return []

    @overrides
    def get_actions(self, observations):
        lb, ub = self.action_bounds
        acts = np.random.random((len(observations), len(lb))) * (ub - lb).reshape(1, -1) + \
                lb.reshape(1, -1)
        return acts, np.ones_like(acts)

    def compute_entropy(self, _):
        return np.nan

