import numpy as np
import sandbox.rocky.tf.core.layers as L
import tensorflow as tf
from sandbox.rocky.tf.core.layers_powered import LayersPowered
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.distributions.diagonal_gaussian import DiagonalGaussian
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.spaces import Box
from sandbox.rocky.tf.policies.base import StochasticPolicy

from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides


class GaussianMLPActorCritic(StochasticPolicy, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.tanh,
            weight_normalization=False,
            layer_normalization=False,
            moments_update_rate=0.9,
    ):
        Serializable.quick_init(self, locals())

        with tf.variable_scope(name):
            assert isinstance(env_spec.action_space, Box)
            super(GaussianMLPActorCritic, self).__init__(env_spec)

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            l_obs = L.InputLayer(
                shape=(None, obs_dim),
                name="obs"
            )

            head_network = MLP(
                name="head_network",
                input_shape=(obs_dim,),
                output_dim=action_dim + 1,
                hidden_sizes=hidden_sizes,
                input_layer=l_obs,
                hidden_nonlinearity=hidden_nonlinearity,
                output_nonlinearity=None,
            )

            l_log_std = L.ParamLayer(
                incoming=l_obs,
                num_units=action_dim,
            )

            self.head_network = head_network
            self.l_obs = l_obs
            self.l_log_std = l_log_std

            self.action_dim = action_dim

            self.moments_update_rate = moments_update_rate

            return_var = tf.placeholder(dtype=tf.float32, shape=(None,), name="return")

            return_mean_var = tf.Variable(
                np.cast['float32'](0.),
                name="return_mean",
            )
            return_std_var = tf.Variable(
                np.cast['float32'](1.),
                name="return_std",
            )
            return_mean_stats = tf.reduce_mean(return_var)
            return_std_stats = tf.sqrt(tf.reduce_mean(tf.square(return_var - return_mean_var)))

            self.return_mean_var = return_mean_var
            self.return_std_var = return_std_var

            self.f_update_stats = tensor_utils.compile_function(
                inputs=[return_var],
                outputs=[
                    tf.assign(
                        return_mean_var,
                        (1 - self.moments_update_rate) * return_mean_var + \
                        self.moments_update_rate * return_mean_stats,
                    ),
                    tf.assign(
                        return_std_var,
                        (1 - self.moments_update_rate) * return_std_var + \
                        self.moments_update_rate * return_std_stats,
                    )
                ]
            )

            self.f_dist = tensor_utils.compile_function(
                inputs=[self.l_obs.input_var],
                outputs=self.dist_info_sym(obs_var=self.l_obs.input_var),
            )

            LayersPowered.__init__(self, [head_network.output_layer])

    def get_params_internal(self, **tags):
        params = LayersPowered.get_params_internal(self, **tags)
        if not tags.get('trainable', False):
            params = params + [self.return_mean_var, self.return_std_var]
        return params

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars=None, **kwargs):
        obs_var = tf.cast(obs_var, tf.float32)
        head_out, log_stds = L.get_output(
            [self.head_network.output_layer, self.l_log_std],
            {self.l_obs: obs_var},
            **kwargs
        )
        means = head_out[:, :self.action_dim]
        vf = head_out[:, self.action_dim:]
        vf = vf * self.return_std_var + self.return_mean_var
        return dict(
            mean=means, log_std=log_stds, vf=vf
        )

    @property
    def vectorized(self):
        return True

    @overrides
    def get_action(self, observation):
        actions, agent_infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in agent_infos.items()}

    @overrides
    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        all_obs = flat_obs
        dist_info = self.f_dist(all_obs)
        action_means = dist_info["mean"]
        vf = dist_info["vf"]
        log_stds = dist_info["log_std"]
        actions = np.random.normal(size=action_means.shape) * np.exp(log_stds) + action_means
        agent_info = dict(mean=action_means, log_std=log_stds, vf=vf)
        return actions, agent_info

    @property
    def distribution(self):
        return DiagonalGaussian(self.action_dim)

    def fit_with_samples(self, paths, samples_data):
        # update the baseline jointly with the policy instead..
        # however, renormalize first
        self.f_update_stats(samples_data["returns"])
