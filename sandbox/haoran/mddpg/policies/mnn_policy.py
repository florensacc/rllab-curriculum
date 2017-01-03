import numpy as np
import tensorflow as tf

from sandbox.haoran.mddpg.core.tf_util import he_uniform_initializer, \
    mlp, linear, weight_variable
from rllab.core.serializable import Serializable
from sandbox.haoran.mddpg.policies.nn_policy import NNPolicy

class MNNPolicy(NNPolicy):
    """ Multi-headed Neural Network Policy """
    def __init__(
        self,
        scope_name,
        observation_dim,
        action_dim,
        K,
        **kwargs
    ):
        Serializable.quick_init(self, locals())
        self.K = K
        self.k = np.random.randint(0,K)
        super(MNNPolicy, self).__init__(
            scope_name, observation_dim, action_dim)

    def create_network(self):
        with tf.variable_scope(self.scope_name):
            with tf.variable_scope('shared'):
                self.shared_variables = self.create_shared_variables()
            self.heads = []
            for k in range(self.K):
                with tf.variable_scope('head_%d'%(k)):
                    self.heads.append(self.create_head(k))

            return tf.pack(self.heads, axis=1, name='outputs')

    def create_head(self, k):
        raise NotImplementedError

    def create_shared_variables(self):
        raise NotImplementedError

    def get_action(self, observation, k=None):
        """
        k: which head to use
        By default, the policy decides which head to use. This is compatible
            with rllab.sampler.utils.rollout(), needed for evaluation.
        k = "all" returns all heads' actions.
        An exploration strategy may overwrite this method and specify particular
            heads.
        """
        if k is None:
            k = self.k
            return self.sess.run(
                self.heads[k],
                {self.observations_placeholder: [observation]}
            ), {
                'heads': k
            }
        elif k == "all":
            return self.sess.run(
                self.output,
                {self.observations_placeholder: [observation]}
            ), {'heads':-1}
        elif (isinstance(k, int) or isinstance(k,np.int64)) and 0 <= k <= self.K:
            return self.sess.run(
                self.heads[k],
                {self.observations_placeholder: [observation]}
            ), {
                'heads': k
            }
        else:
            raise NotImplementedError



    def get_actions(self, observations):
        """
        By default, returns all candidate actions, since this method is probably
        only used when updating Q and pi.
        """
        return self.sess.run(
            self.output,
            {self.observations_placeholder: observations}
        ), {'heads': -np.ones(len(observations))}

    def reset(self):
        self.k = np.random.randint(0,self.K)


from rllab.exploration_strategies.base import ExplorationStrategy
class MNNStrategy(ExplorationStrategy):
    """
    Cleverly chooses between heads to do exploration.
    The current version switches a head after finishing a traj.
    """
    def __init__(self, K, substrategy, switch_type):
        self.K = K
        self.substrategy = substrategy
        self.switch_type = switch_type

        assert self.switch_type in ["per_action", "per_path"]

        self.k = 0 # current head

    def get_action(self, t, observation, policy, **kwargs):
        assert isinstance(policy, MNNPolicy)
        action, _ = policy.get_action(observation, self.k)
        action_modified = self.substrategy.get_modified_action(t, action)
        if self.switch_type == "per_action":
            self.k = np.mod(self.k + 1, self.K)
            # print("{} switches to head {}".format(policy.scope_name, self.k))
        return action_modified

    def reset(self):
        if self.switch_type == "per_path":
            self.k = np.mod(self.k + 1, self.K)
        self.substrategy.reset()

class FeedForwardMultiPolicy(MNNPolicy):
    def __init__(
        self,
        scope_name,
        observation_dim,
        action_dim,
        K,
        shared_hidden_sizes=(100, 100),
        independent_hidden_sizes=tuple(),
        hidden_W_init=None,
        hidden_b_init=None,
        output_W_init=None,
        output_b_init=None,
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh,
    ):
        Serializable.quick_init(self, locals())
        self.shared_hidden_sizes = shared_hidden_sizes
        self.independent_hidden_sizes = independent_hidden_sizes
        self.hidden_W_init = hidden_W_init or he_uniform_initializer()
        self.hidden_b_init = hidden_b_init or tf.constant_initializer(0.)
        self.output_W_init = output_W_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.output_b_init = output_b_init or tf.random_uniform_initializer(
            -3e-3, 3e-3)
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        super(FeedForwardMultiPolicy, self).__init__(
            scope_name,
            observation_dim,
            action_dim,
            K,
        )
    def create_shared_variables(self):
        shared_layer = mlp(
            self.observations_placeholder,
            self.observation_dim,
            self.shared_hidden_sizes,
            self.hidden_nonlinearity,
            W_initializer=self.hidden_W_init,
            b_initializer=self.hidden_b_init,
        )
        return {"shared_layer": shared_layer}

    def create_head(self,k):
        if len(self.shared_hidden_sizes) > 0:
            shared_output_size = self.shared_hidden_sizes[-1]
        else:
            shared_output_size = self.observation_dim
        preoutput_layer = mlp(
            self.shared_variables["shared_layer"],
            shared_output_size,
            self.independent_hidden_sizes,
            self.hidden_nonlinearity,
            W_initializer=self.hidden_W_init,
            b_initializer=self.hidden_b_init,
        )
        if len(self.independent_hidden_sizes) > 0:
            preoutput_layer_size = self.independent_hidden_sizes[-1]
        elif len(self.shared_hidden_sizes) > 0:
            preoutput_layer_size = self.shared_hidden_sizes[-1]
        else:
            preoutput_layer_size = self.observation_dim
        output = self.output_nonlinearity(linear(
            preoutput_layer,
            preoutput_layer_size,
            self.action_dim,
            W_initializer=self.output_W_init,
            b_initializer=self.output_b_init,
        ))
        return output
