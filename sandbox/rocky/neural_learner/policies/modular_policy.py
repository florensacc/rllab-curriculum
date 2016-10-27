import numpy as np
import tensorflow as tf
from cached_property import cached_property

from sandbox.rocky.tf.core.parameterized import Parameterized
from sandbox.rocky.tf.distributions.recurrent_categorical import RecurrentCategorical
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.policies.base import StochasticPolicy

from rllab.core.serializable import Serializable
from rllab.misc import special


class ModularPolicy(StochasticPolicy, Parameterized, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            net,
    ):
        Serializable.quick_init(self, locals())

        with tf.variable_scope(name):
            prob_network, = net.new_networks(env_spec=env_spec)
            self.prob_network = prob_network

            obs_var = env_spec.observation_space.new_tensor_variable(
                "obs",
                extra_dims=1
            )
            dones_var = tf.placeholder(tf.float32, (None,), "dones")

            self.f_step_prob = tensor_utils.compile_function(
                [obs_var],
                prob_network.get_step_op(
                    obs_var=obs_var,
                    phase='test'
                )
            )
            self.f_full_reset = tensor_utils.compile_function(
                [dones_var],
                prob_network.get_full_reset_op(dones_var),
            )
            self.f_partial_reset = tensor_utils.compile_function(
                [dones_var],
                prob_network.get_partial_reset_op(dones_var),
            )
            self.prev_reset_length = -1

    def get_params_internal(self, **tags):
        return self.prob_network.get_params(**tags)

    def dist_info_sym(self, obs_var, state_info_vars, **kwargs):
        obs_var = tf.cast(obs_var, tf.float32)
        return dict(
            prob=self.prob_network.get_output(obs_var=obs_var, **kwargs)
        )

    @property
    def vectorized(self):
        return True

    def reset(self, dones=None):
        if dones is None:
            dones = [True]
        if len(dones) != self.prev_reset_length:
            self.f_full_reset(dones)
        else:
            self.f_partial_reset(dones)
        self.prev_reset_length = len(dones)

    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        probs = self.f_step_prob(flat_obs)
        actions = special.weighted_sample_n(probs, np.arange(self.action_space.n))
        agent_info = dict(prob=probs)
        return actions, agent_info

    @property
    def recurrent(self):
        return True

    @cached_property
    def distribution(self):
        return RecurrentCategorical(self.action_space.n)

    @property
    def state_info_specs(self):
        return []
