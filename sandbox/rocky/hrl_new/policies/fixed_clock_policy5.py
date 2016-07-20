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

            l_bottleneck_obs = L.SliceLayer(
                l_obs,
                indices=slice(env_spec.observation_space.components[0].flat_dim),
                name="bottleneck_obs"
            )

            self.bottleneck_network = MLP(
                input_shape=(obs_dim - env_spec.observation_space.components[0].flat_dim,),
                input_layer=l_bottleneck_obs,
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
            self.subgoal_obs = None

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
            self.subgoal_obs = np.zeros((len(dones), self.obs_dim))
        self.ts[dones] = -1
        self.subgoals[dones] = np.nan
        self.subgoal_probs[dones] = np.nan
        self.subgoal_obs[dones] = np.nan

    def get_subgoals(self, flat_obs):
        probs = self.f_subgoal_prob(flat_obs)
        subgoals = np.cast['int'](weighted_sample_n(probs, np.arange(self.subgoal_dim)))
        return subgoals, probs

    def get_bottlenecks(self, flat_obs):
        probs = self.f_bottleneck_prob(flat_obs)
        bottlenecks = np.cast['int'](weighted_sample_n(probs, np.arange(self.bottleneck_dim)))
        return bottlenecks, probs

    @property
    def distribution(self):
        return Categorical(self.action_dim)

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
        subgoal_prob = L.get_output(self.l_subgoal_prob, {self.l_obs: state_info_vars["subgoal_obs"]})
        all_action_prob = self.get_all_action_prob_sym(obs_var)
        N = tf.shape(all_action_prob)[0]

        grouped_action_probs = tf.reshape(
            all_action_prob,
            (N / self.subgoal_interval, self.subgoal_interval, self.subgoal_dim, self.action_dim)
        )
        # shape: (N/K)*K*|G|*|A|
        grouped_action_probs.set_shape((None, self.subgoal_interval, self.subgoal_dim, self.action_dim))

        grouped_joint_action_probs = self.outer_probs(grouped_action_probs)

        grouped_subgoal_prob = tf.reshape(
            subgoal_prob,
            (N / self.subgoal_interval, self.subgoal_interval, self.subgoal_dim)
        )

        grouped_subgoal_prob.set_shape((None, self.subgoal_interval, self.subgoal_dim))

        # just take the first one, since it's supposed to be the same
        grouped_joint_subgoal_prob = tf.reshape(
            grouped_subgoal_prob[:, 0, :],
            (-1, self.subgoal_dim) + (1, ) * self.subgoal_interval
        )

        marginal_joint_action_probs = tf.reduce_sum(grouped_joint_subgoal_prob * grouped_joint_action_probs, 1)

        return dict(joint_action_probs=marginal_joint_action_probs)

    def likelihood_ratio_sym(self, action_var, old_dist_info_vars, new_dist_info_vars):
        action_var = tf.cast(action_var, tf.float32)
        N = tf.shape(action_var)[0]
        action_var = tf.reshape(action_var, (N / self.subgoal_interval, self.subgoal_interval, 1, self.action_dim))
        action_var.set_shape((None, self.subgoal_interval, 1, self.action_dim))
        action_dist = self.outer_probs(action_var)
        action_dist = tf.reshape(action_dist, (N / self.subgoal_interval,) + (self.action_dim,) * self.subgoal_interval)
        new_joint_probs = new_dist_info_vars["joint_action_probs"]
        old_joint_probs = old_dist_info_vars["joint_action_probs"]
        new_prob = tf.reduce_sum(tf.reshape(action_dist * new_joint_probs, tf.pack([N / self.subgoal_interval, -1])),
                                 -1)
        old_prob = tf.reduce_sum(tf.reshape(action_dist * old_joint_probs, tf.pack([N / self.subgoal_interval, -1])),
                                 -1)
        return new_prob / (old_prob + TINY)

    def entropy(self, obs, valids, state_infos):
        dist_infos = self.dist_info(obs, state_infos)
        probs = dist_infos["joint_action_probs"]
        probs = probs.reshape((-1, self.action_dim ** self.subgoal_interval))
        ents = np.sum(-probs * np.log(probs), axis=-1)
        valids = valids.reshape((-1, self.subgoal_interval)).prod(axis=-1)
        return np.sum(ents * valids) / np.sum(valids) / self.subgoal_interval

    def final_outer(self, x, y):
        ndim = x.get_shape().ndims
        x_static_shape = x.get_shape().as_list()
        y_static_shape = y.get_shape().as_list()
        prior_shape = tf.shape(x)[:ndim - 1]
        last_shape_x = tf.shape(x)[ndim - 1]
        last_shape_y = tf.shape(y)[ndim - 1]
        flat_x = tf.reshape(x, tf.pack([-1, last_shape_x]))
        flat_y = tf.reshape(y, tf.pack([-1, last_shape_y]))
        flat_outer = tf.expand_dims(flat_x, -1) * tf.expand_dims(flat_y, 1)
        reshaped_outer = tf.reshape(flat_outer, tf.concat(0, [prior_shape, tf.shape(flat_outer)[1:]]))
        static_prior_shape = x_static_shape[:-1]
        static_last_shape_x = x_static_shape[-1]
        static_last_shape_y = y_static_shape[-1]
        reshaped_outer.set_shape(static_prior_shape + [static_last_shape_x, static_last_shape_y])
        return reshaped_outer
        # flat_x = tf.reshape(x, tf.pack([-1, last_shape])):weighted_sample_n()

    def outer_probs(self, input_probs):
        # shape: (N/K)*K*|G|*|A|
        # assert input_probs.get_shape().as_list()[1] == 3

        idx = 0
        prob = input_probs[:, idx, :, :]

        N = tf.shape(input_probs)[0]
        subgoal_dim = tf.shape(input_probs)[2]
        static_subgoal_dim = input_probs.get_shape().as_list()[2]

        for next_idx in xrange(1, self.subgoal_interval):
            next_prob = input_probs[:, next_idx, :, :]

            reshaped_prob = tf.reshape(prob, tf.pack([N, subgoal_dim, -1]))

            # prob_0 = input_probs[:, 0, :, :]
            # prob_1 = input_probs[:, 1, :, :]
            # prob_2 = input_probs[:, 2, :, :]

            x = self.final_outer(reshaped_prob, next_prob)
            x = tf.reshape(x, tf.pack([N, subgoal_dim] + [self.action_dim] * (next_idx + 1)))
            prob = x

        prob.set_shape((None, static_subgoal_dim) + (self.action_dim,) * self.subgoal_interval)
        # import ipdb;
        # ipdb.set_trace()

        # x = tf.reshape(x, tf.concat(0, [prior_shape, [-1]]))
        # x.set_shape(static_prior_shape + [self.action_dim * self.action_dim])
        # x = self.final_outer(x, prob_2)
        # x = tf.reshape(x, tf.concat(0, [prior_shape, [self.action_dim, self.action_dim, self.action_dim]]))
        # x.set_shape(static_prior_shape + [self.action_dim, self.action_dim, self.action_dim])
        return prob

    def proper_mean_kl_sym_flat_vs_mixture(self, obs_var, valid_var, state_info_vars, old_dist_info_vars):

        assert self.subgoal_interval == 3

        valid_var = tf.cast(valid_var, tf.float32)
        # return tf.reduce_mean(self.distribution.kl_sym(old_dist_info_vars, self.dist_info_sym(obs_var,
        #                                                                                       state_info_vars)))
        subgoal_prob = L.get_output(self.l_subgoal_prob, {self.l_obs: state_info_vars["subgoal_obs"]})
        all_action_prob = self.get_all_action_prob_sym(obs_var)
        N = tf.shape(all_action_prob)[0]

        grouped_action_probs = tf.reshape(
            all_action_prob,
            (N / self.subgoal_interval, self.subgoal_interval, self.subgoal_dim, self.action_dim)
        )
        # shape: (N/K)*K*|G|*|A|
        grouped_action_probs.set_shape((None, self.subgoal_interval, self.subgoal_dim, self.action_dim))

        grouped_joint_action_probs = self.outer_probs(grouped_action_probs)

        old_probs = old_dist_info_vars["prob"]

        old_grouped_probs = tf.reshape(
            old_probs,
            (N / self.subgoal_interval, self.subgoal_interval, self.action_dim)
        )

        old_grouped_probs.set_shape((None, self.subgoal_interval, self.action_dim))
        # shape: (N/K)*K*1*|A|
        old_grouped_probs = tf.expand_dims(old_grouped_probs, 2)

        old_grouped_joint_probs = self.outer_probs(old_grouped_probs)

        grouped_subgoal_prob = tf.reshape(
            subgoal_prob,
            (N / self.subgoal_interval, self.subgoal_interval, self.subgoal_dim)
        )

        grouped_subgoal_prob.set_shape((None, self.subgoal_interval, self.subgoal_dim))

        # just take the first one, since it's supposed to be the same
        grouped_joint_subgoal_prob = tf.reshape(grouped_subgoal_prob[:, 0, :], (-1, self.subgoal_dim, 1, 1, 1))

        marginal_joint_action_probs = tf.reduce_sum(grouped_joint_subgoal_prob * grouped_joint_action_probs, 1,
                                                    keep_dims=True)

        # import ipdb; ipdb.set_trace()
        kl_sym = old_grouped_joint_probs * (tf.log(old_grouped_joint_probs + TINY) - tf.log(
            marginal_joint_action_probs + TINY))
        kl_sym = tf.reduce_sum(tf.reshape(kl_sym, [-1, self.action_dim * self.action_dim * self.action_dim]), -1)

        grouped_valid_var = tf.reduce_prod(tf.reshape(valid_var, (-1, 3)), -1)

        kl_sym = tf.reduce_sum(kl_sym * grouped_valid_var) / tf.reduce_sum(grouped_valid_var) / self.subgoal_interval

        return kl_sym

    def proper_mean_kl_sym_mixture_vs_flat(self, obs_var, valid_var, state_info_vars, old_dist_info_vars):
        assert self.subgoal_interval == 3

        valid_var = tf.cast(valid_var, tf.float32)
        # return tf.reduce_mean(self.distribution.kl_sym(old_dist_info_vars, self.dist_info_sym(obs_var,
        #                                                                                       state_info_vars)))
        subgoal_prob = L.get_output(self.l_subgoal_prob, {self.l_obs: state_info_vars["subgoal_obs"]})
        all_action_prob = self.get_all_action_prob_sym(obs_var)
        N = tf.shape(all_action_prob)[0]

        grouped_action_probs = tf.reshape(
            all_action_prob,
            (N / self.subgoal_interval, self.subgoal_interval, self.subgoal_dim, self.action_dim)
        )
        # shape: (N/K)*K*|G|*|A|
        grouped_action_probs.set_shape((None, self.subgoal_interval, self.subgoal_dim, self.action_dim))

        grouped_joint_action_probs = self.outer_probs(grouped_action_probs)

        old_probs = old_dist_info_vars["prob"]

        old_grouped_probs = tf.reshape(
            old_probs,
            (N / self.subgoal_interval, self.subgoal_interval, self.action_dim)
        )

        old_grouped_probs.set_shape((None, self.subgoal_interval, self.action_dim))
        # shape: (N/K)*K*1*|A|
        old_grouped_probs = tf.expand_dims(old_grouped_probs, 2)

        old_grouped_joint_probs = self.outer_probs(old_grouped_probs)

        grouped_subgoal_prob = tf.reshape(
            subgoal_prob,
            (N / self.subgoal_interval, self.subgoal_interval, self.subgoal_dim)
        )

        grouped_subgoal_prob.set_shape((None, self.subgoal_interval, self.subgoal_dim))

        # just take the first one, since it's supposed to be the same
        grouped_joint_subgoal_prob = tf.reshape(grouped_subgoal_prob[:, 0, :], (-1, self.subgoal_dim, 1, 1, 1))

        marginal_joint_action_probs = tf.reduce_sum(grouped_joint_subgoal_prob * grouped_joint_action_probs, 1,
                                                    keep_dims=True)

        # import ipdb; ipdb.set_trace()
        kl_sym = marginal_joint_action_probs * (tf.log(marginal_joint_action_probs + TINY) - tf.log(
            old_grouped_joint_probs + TINY))

        kl_sym = tf.reduce_sum(tf.reshape(kl_sym, [-1, self.action_dim * self.action_dim * self.action_dim]), -1)

        grouped_valid_var = tf.reduce_prod(tf.reshape(valid_var, (-1, 3)), -1)
        self.f_debug = tensor_utils.compile_function(
            [obs_var, valid_var] + [state_info_vars[k] for k in self.state_info_keys] + [old_dist_info_vars[k] for k
                                                                                         in
                                                                                         self.distribution.dist_info_keys],
            [marginal_joint_action_probs, old_grouped_joint_probs, kl_sym, grouped_valid_var]
        )
        kl_sym = tf.reduce_sum(kl_sym * grouped_valid_var) / tf.reduce_sum(grouped_valid_var) / self.subgoal_interval

        return kl_sym

    @property
    def state_info_specs(self):
        return [
            ("subgoal_obs", (self.obs_dim,)),
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
            ("joint_action_probs", (self.action_dim,) * self.subgoal_interval),
        ]

    @property
    def dist_info_keys(self):
        return [x for x, _ in self.dist_info_specs]

    def kl_sym(self, old_dist_info_vars, dist_info_vars):
        old_joint_probs = old_dist_info_vars["joint_action_probs"]
        new_joint_probs = dist_info_vars["joint_action_probs"]
        kl_sym = old_joint_probs * (tf.log(old_joint_probs + TINY) - tf.log(new_joint_probs + TINY))
        kl_sym = tf.reduce_sum(tf.reshape(kl_sym, [-1, self.action_dim ** self.subgoal_interval]), -1)
        return kl_sym

    def get_actions(self, observations):
        self.ts += 1
        flat_obs = self.observation_space.flatten_n(observations)
        subgoals, subgoal_probs = self.get_subgoals(flat_obs)
        update_mask = self.ts % self.subgoal_interval == 0
        self.subgoals[update_mask] = self.subgoal_space.flatten_n(subgoals[update_mask])
        self.subgoal_probs[update_mask] = subgoal_probs[update_mask]
        self.subgoal_obs[update_mask] = flat_obs[update_mask]

        # instead of explicitly sampling bottlenecks, we directly marginalize over the distribution to get p(a|g)
        action_probs = self.f_action_prob(flat_obs, self.subgoals)

        actions = weighted_sample_n(action_probs, np.arange(self.action_dim))

        # compute the marginal p(a|s), where the marginalization is over all subgoals

        return actions, dict(
            # prob=action_marginal_prob,
            subgoal_obs=np.copy(self.subgoal_obs),
        )
