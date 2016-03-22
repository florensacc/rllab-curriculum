from rllab.policy.base import StochasticPolicy
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.policy.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policy.mean_std_nn_policy import MeanStdNNPolicy
from rllab.misc import autoargs
from rllab.misc.ext import AttrDict, flatten_shape_dim
from rllab.misc.special import to_onehot
import numpy as np


class DiscreteSubgoalGaussianPolicy(StochasticPolicy, LasagnePowered,
                                    Serializable):

    """
    The high-level policy receives the raw observation, and emits a subgoal
    for the low-level policy. The low-level policy receives the raw observation
    concatenated with the subgoal, and emits the actual control for the MDP.
    """

    @autoargs.arg('high_hidden_sizes', type=int, nargs='*',
                  help='hidden layer sizes for the high-level policy')
    @autoargs.arg('high_nonlinearity', type=str,
                  help='nonlinearity for the high-level policy')
    @autoargs.arg('low_hidden_sizes', type=int, nargs='*',
                  help='hidden layer sizes for the high-level policy')
    @autoargs.arg('low_nonlinearity', type=str,
                  help='nonlinearity for the high-level policy')
    def __init__(
            self,
            mdp,
            n_goals=10,
            high_hidden_sizes=(32, 32),
            high_nonlinearity='lasagne.nonlinearities.rectify',
            low_hidden_sizes=(32, 32),
            low_nonlinearity='lasagne.nonlinearities.rectify'):
        Serializable.quick_init(self, locals())
        super(DiscreteSubgoalGaussianPolicy, self).__init__(mdp)
        self._n_goals = n_goals
        # Here, we have to pass in a dummy mdp encoding the information
        self._high_policy = CategoricalMLPPolicy(
            mdp=AttrDict(
                observation_shape=mdp.observation_shape,
                observation_dtype=mdp.observation_dtype,
                action_dim=n_goals,
                action_dtype='uint8'
            ),
            hidden_sizes=high_hidden_sizes,
            nonlinearity=high_nonlinearity
        )
        obs_dim = flatten_shape_dim(mdp.observation_shape)
        self._low_policy = MeanStdNNPolicy(
            mdp=AttrDict(
                observation_shape=(obs_dim + n_goals,),
                observation_dtype=mdp.observation_dtype,
            ),
            hidden_sizes=low_hidden_sizes,
            nonlinearity=low_nonlinearity
        )

    @property
    def high_policy(self):
        return self._high_policy

    @property
    def low_policy(self):
        return self._low_policy

    def act(self, observation):
        # First, sample a goal
        goal, high_pdist = self._high_policy.act(observation)
        goal_onehot = to_onehot(goal, self._n_goals)
        action, low_pdist = self._low_policy.act(np.concatenate([
            observation.flatten(),
            goal_onehot
        ]))
        return action, np.concatenate([high_pdist, low_pdist])
