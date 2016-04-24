from __future__ import print_function
from __future__ import absolute_import

from rllab.policies.base import Policy
import lasagne.nonlinearities as NL
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.policies.base import StochasticPolicy
from rllab.core.network import MLP
from sandbox.rocky.hrl.core.network import MergeMLP
from rllab.spaces.discrete import Discrete
from rllab.distributions.categorical import Categorical
from rllab.misc import ext
from rllab.misc.overrides import overrides
import lasagne.layers as L


class StateGoalCategoricalMLPPolicy(StochasticPolicy, LasagnePowered, Serializable):
    def __init__(
            self,
            env_spec,
            subgoal_space,
            state_hidden_sizes=(32,),
            goal_hidden_sizes=(32,),
            joint_hidden_sizes=(32,),
            hidden_nonlinearity=NL.tanh,
    ):
        Serializable.quick_init(self, locals())

        assert isinstance(env_spec.action_space, Discrete)

        goal_dim = subgoal_space.flat_dim
        obs_dim = env_spec.observation_space.flat_dim - goal_dim
        action_dim = env_spec.action_space.flat_dim
        prob_network = MergeMLP(
            input_shape=(obs_dim + goal_dim,),
            branch_dims=[obs_dim, goal_dim],
            output_dim=action_dim,
            branch_hidden_sizes=[state_hidden_sizes, goal_hidden_sizes],
            joint_hidden_sizes=joint_hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=NL.softmax,
        )

        self._l_in = prob_network.input_layer
        self._l_prob = prob_network.output_layer
        self._f_prob = ext.compile_function([prob_network.input_var], prob_network.output)

        self._dist = Categorical()
        super(StateGoalCategoricalMLPPolicy, self).__init__(env_spec)
        LasagnePowered.__init__(self, [prob_network.output_layer])

    @overrides
    def dist_info_sym(self, obs_var, action_var):
        return dict(prob=L.get_output(self._l_prob, {self._l_in: obs_var}))

    @overrides
    def dist_info(self, obs, actions):
        return dict(prob=self._f_prob(obs))

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        prob = self._f_prob([flat_obs])[0]
        action = self.action_space.weighted_sample(prob)
        return action, dict(prob=prob)

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        probs = self._f_prob(flat_obs)
        actions = map(self.action_space.weighted_sample, probs)
        return actions, dict(prob=probs)

    @property
    def distribution(self):
        return self._dist
