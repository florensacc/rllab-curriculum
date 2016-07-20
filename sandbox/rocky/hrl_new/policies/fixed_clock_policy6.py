from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.tf.policies.base import StochasticPolicy
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.spaces.discrete import Discrete
from sandbox.rocky.tf.distributions.categorical import Categorical
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.misc import tensor_utils
import sandbox.rocky.tf.core.layers as L
import tensorflow as tf
import numpy as np

"""

"""

TINY = 1e-8


def weighted_sample_n(prob_matrix, items):
    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0])
    k = (s < r.reshape((-1, 1))).sum(axis=1)
    n_items = len(items)
    return items[np.minimum(k, n_items - 1)]


class FixedClockPolicy(StochasticPolicy, Serializable):
    def __init__(
            self,
            env_spec,
            name,
            subgoal_dim,
            bottleneck_dim,
            subgoal_interval,
            hidden_sizes=(32, 32),
            log_prob_tensor_std=1.0,
    ):

        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.action_space, Discrete)

        with tf.variable_scope(name):
            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            self.subgoal_network = MLP(
                input_shape=(obs_dim,),
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=tf.nn.tanh,
                output_dim=subgoal_dim,
                output_nonlinearity=tf.nn.softmax,
                name="subgoal_network"
            )

            l_obs = self.subgoal_network.input_layer
            l_subgoal_prob = self.subgoal_network.output_layer
            obs_var = l_obs.input_var

            # l_bottleneck_obs = L.SliceLayer(
            #     l_obs,
            #     indices=slice(env_spec.observation_space.components[0].flat_dim),
            #     name="bottleneck_obs"
            # )

            self.bottleneck_network = MLP(
                input_shape=(obs_dim,),#(obs_dim - env_spec.observation_space.components[0].flat_dim,),
                input_layer=l_obs,
                hidden_sizes=hidden_sizes,
                hidden_nonlinearity=tf.nn.tanh,
                output_dim=bottleneck_dim,
                output_nonlinearity=tf.nn.softmax,
                name="bottleneck_network",
            )

            l_bottleneck_prob = self.bottleneck_network.output_layer

            log_prob_tensor = tf.Variable(
                initial_value=np.cast['float32'](
                    np.random.normal(scale=log_prob_tensor_std, size=(bottleneck_dim, subgoal_dim, action_dim))
                ),
                trainable=True,
                name="log_prob"
            )

            prob_tensor = tf.reshape(
                tf.nn.softmax(tf.reshape(log_prob_tensor, (-1, action_dim))),
                (bottleneck_dim, subgoal_dim, action_dim)
            )
            subgoal_space = Discrete(subgoal_dim)
            bottleneck_space = Discrete(bottleneck_dim)

            # record current execution time steps
            self.ts = None
            # record current subgoals
            self.subgoals = None
            self.subgoal_probs = None

            self.subgoal_space = subgoal_space
            self.bottleneck_space = bottleneck_space
            self.subgoal_dim = subgoal_dim
            self.bottleneck_dim = bottleneck_dim
            self.obs_dim = obs_dim
            self.action_dim = action_dim
            self.l_obs = l_obs
            self.l_bottleneck_prob = l_bottleneck_prob
            self.l_subgoal_prob = l_subgoal_prob
            self.log_prob_tensor = log_prob_tensor
            self.prob_tensor = prob_tensor
            self.subgoal_interval = subgoal_interval
            self.subgoal_dist = Categorical(subgoal_dim)
            self.action_dist = Categorical(action_dim)
            self.bottleneck_dist = Categorical(bottleneck_dim)

            StochasticPolicy.__init__(self, env_spec)

            self.f_subgoal_prob = tensor_utils.compile_function(
                inputs=[obs_var],
                outputs=L.get_output(l_subgoal_prob),
            )
            self.f_bottleneck_prob = tensor_utils.compile_function(
                inputs=[obs_var],
                outputs=L.get_output(l_bottleneck_prob),
            )
            subgoal_var = tf.placeholder(
                dtype=tf.float32,
                shape=(None, self.subgoal_dim),
                name="subgoal",

            )
            self.f_action_prob = tensor_utils.compile_function(
                inputs=[obs_var, subgoal_var],
                outputs=self.get_action_prob_sym(obs_var, subgoal_var),
            )

    @property
    def vectorized(self):
        return True

    def get_params_internal(self, **tags):
        return self.subgoal_network.get_params_internal(**tags) + self.bottleneck_network.get_params_internal(**tags) \
               + [self.log_prob_tensor]

    def get_action(self, observation):
        actions, infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in infos.iteritems()}

    def reset(self, dones=None):
        if dones is None:
            dones = [True]
        dones = np.asarray(dones)
        if self.ts is None or len(dones) != len(self.ts):
            self.ts = np.array([-1] * len(dones))
            self.subgoals = np.zeros((len(dones), self.subgoal_dim))
            self.subgoal_probs = np.zeros((len(dones), self.subgoal_dim))
        self.ts[dones] = -1
        self.subgoals[dones] = np.nan
        self.subgoal_probs[dones] = np.nan

    def get_subgoals(self, flat_obs):
        probs = self.f_subgoal_prob(flat_obs)
        subgoals = np.cast['int'](weighted_sample_n(probs, np.arange(self.subgoal_dim)))
        return subgoals, probs

    def get_bottlenecks(self, flat_obs):
        probs = self.f_bottleneck_prob(flat_obs)
        bottlenecks = np.cast['int'](weighted_sample_n(probs, np.arange(self.bottleneck_dim)))
        return bottlenecks, probs

    def get_action_prob_sym(self, obs_var, subgoal_var):
        # subgoal_var can be either one-hot or probabilities
        # N*|Z|
        subgoal_var = tf.cast(subgoal_var, tf.float32)
        x = self.get_all_action_prob_sym(obs_var)
        # result: N*|A|
        x = tf.reduce_sum(tf.expand_dims(subgoal_var, -1) * x, 1)
        return x

    def get_all_action_prob_sym(self, obs_var):
        bottleneck_prob = L.get_output(self.l_bottleneck_prob, {self.l_obs: obs_var})
        # result: N*(|G|*|A|)
        x = tf.matmul(bottleneck_prob, tf.reshape(self.prob_tensor, (self.bottleneck_dim, -1)))
        # result: N*|G|*|A|
        x = tf.reshape(x, (-1, self.subgoal_dim, self.action_dim))
        return x

    def dist_info_sym(self, obs_var, state_info_vars):
        subgoal_prob = L.get_output(self.l_subgoal_prob, {self.l_obs: obs_var})
        action_prob = self.get_action_prob_sym(obs_var, state_info_vars["subgoal"])
        return dict(
            action_prob=action_prob,
            subgoal_prob=subgoal_prob,
            update_mask=tf.identity(state_info_vars["update_mask"]),
            subgoal=tf.identity(state_info_vars["subgoal"]),
        )

    def likelihood_ratio_sym(self, action_var, old_dist_info_vars, new_dist_info_vars):
        subgoal_var = old_dist_info_vars["subgoal"]
        update_mask = old_dist_info_vars["update_mask"]
        old_action_prob = old_dist_info_vars["action_prob"]
        old_subgoal_prob = old_dist_info_vars["subgoal_prob"]
        new_action_prob = new_dist_info_vars["action_prob"]
        new_subgoal_prob = new_dist_info_vars["subgoal_prob"]

        action_lr = self.action_dist.likelihood_ratio_sym(action_var, dict(prob=old_action_prob),
                                                          dict(prob=new_action_prob))
        subgoal_lr = self.subgoal_dist.likelihood_ratio_sym(subgoal_var, dict(prob=old_subgoal_prob),
                                                            dict(prob=new_subgoal_prob))

        subgoal_lr = subgoal_lr * update_mask + (1. - update_mask)
        return action_lr * subgoal_lr

    def entropy(self, dist_info):
        return self.action_dist.entropy(dict(prob=dist_info["action_prob"]))

    @property
    def state_info_specs(self):
        return [
            ("update_mask", (1,)),
            ("subgoal", (self.subgoal_dim,)),
        ]

    def dist_info(self, obs, state_infos):
        if not hasattr(self, "f_dist"):
            obs_var = self.l_obs.input_var
            state_info_vars_list = [
                tf.placeholder(tf.float32, (None,) + shape, name=k) for k, shape in self.state_info_specs
                ]
            state_info_vars = dict(zip(self.state_info_keys, state_info_vars_list))
            self.f_dist = tensor_utils.compile_function(
                inputs=[obs_var] + state_info_vars_list,
                outputs=self.dist_info_sym(obs_var, state_info_vars)
            )
        return self.f_dist(obs, *[state_infos[k] for k in self.state_info_keys])

    @property
    def distribution(self):
        return self

    @property
    def dist_info_specs(self):
        return [
            ("update_mask", (1,)),
            ("subgoal", (self.subgoal_dim,)),
            ("subgoal_prob", (self.subgoal_dim,)),
            ("action_prob", (self.action_dim,)),
        ]

    @property
    def dist_info_keys(self):
        return [x for x, _ in self.dist_info_specs]

    def kl_sym(self, old_dist_info_vars, dist_info_vars):
        old_action_prob = old_dist_info_vars["action_prob"]
        old_subgoal_prob = old_dist_info_vars["subgoal_prob"]
        new_action_prob = dist_info_vars["action_prob"]
        new_subgoal_prob = dist_info_vars["subgoal_prob"]
        update_mask = old_dist_info_vars["update_mask"]

        action_kl = self.action_dist.kl_sym(dict(prob=old_action_prob), dict(prob=new_action_prob))
        subgoal_kl = self.subgoal_dist.kl_sym(dict(prob=old_subgoal_prob), dict(prob=new_subgoal_prob))

        subgoal_kl = subgoal_kl * update_mask

        return action_kl + subgoal_kl

    def get_actions(self, observations):
        self.ts += 1
        flat_obs = self.observation_space.flatten_n(observations)
        subgoals, subgoal_probs = self.get_subgoals(flat_obs)
        update_mask = self.ts % self.subgoal_interval == 0
        self.subgoals[update_mask] = self.subgoal_space.flatten_n(subgoals[update_mask])
        self.subgoal_probs[update_mask] = subgoal_probs[update_mask]

        # instead of explicitly sampling bottlenecks, we directly marginalize over the distribution to get p(a|g)
        action_probs = self.f_action_prob(flat_obs, self.subgoals)

        actions = weighted_sample_n(action_probs, np.arange(self.action_dim))

        return actions, dict(
            subgoal_prob=np.copy(self.subgoal_probs),
            subgoal=np.copy(self.subgoals),
            action_prob=action_probs,
            update_mask=np.reshape(np.cast['int'](update_mask), (-1, 1)),
        )
