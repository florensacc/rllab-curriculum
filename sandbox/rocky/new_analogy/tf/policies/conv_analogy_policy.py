from cached_property import cached_property
from rllab.core.serializable import Serializable
import numpy as np

import keras.layers as L
from keras.models import Model

from sandbox.rocky.new_analogy import keras_ext
from sandbox.rocky.tf.core.keras_powered import KerasPowered

keras_ext.inject()

from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.regressors.auto_mlp_regressor import space_to_dist_dim, output_to_info, space_to_distribution
import tensorflow as tf
import keras.backend as K


def maybe_norm(batch_norm, activation):
    def _apply(layer):
        if batch_norm:
            layer = L.BatchNormalization()(layer)
        layer = L.Activation(activation)(layer)
        return layer

    return _apply


class ConvAnalogyPolicy(KerasPowered, Serializable):
    def __init__(
            self,
            env_spec,
            activation='relu',
            dilations=[1, 2, 4, 8, 16, 32, 64, 128] * 5,
            batch_norm=False,
    ):
        Serializable.quick_init(self, locals())
        obs_dim = env_spec.observation_space.flat_dim
        action_flat_dim = space_to_dist_dim(env_spec.action_space)
        self.env_spec = env_spec
        self.observation_space = env_spec.observation_space
        self.action_space = env_spec.action_space

        N = maybe_norm(batch_norm=batch_norm, activation=activation)

        demo_obs = L.Input(shape=(None, obs_dim))

        def _new_demo_embedding():
            h = N(L.Conv1D(64, 1, border_mode='same')(demo_obs))
            for dilation in dilations:  # [1, 2, 4]:#, 8, 16, 32, 64, 128, 256]:
                h_down = N(L.Conv1D(32, 1, border_mode='same')(h))
                transformed = N(L.AtrousConv1D(32, 2, border_mode='same', atrous_rate=dilation)(h_down))
                h_up = N(L.Conv1D(64, 1, border_mode='same')(transformed))
                h = L.merge([h, h_up], mode='sum')
            return N(L.Conv1D(64, 1, border_mode='same')(h))

        demo_embedding = _new_demo_embedding()

        policy_obs = L.Input(shape=(None, obs_dim))

        def _new_joint_embedding():
            h = N(L.Conv1D(64, 1, border_mode='same')(policy_obs))
            h = N(L.merge([h, demo_embedding], mode='concat'))
            h = N(L.Conv1D(64, 1, border_mode='same')(h))
            return h

        joint_embedding = _new_joint_embedding()

        def _new_decoder_output():
            h = joint_embedding
            # start being causal..
            for dilation in dilations:  # [1, 2, 4]:#, 8, 16, 32, 64, 128, 256]:
                h_down = N(L.Conv1D(32, 1, border_mode='same')(h))
                transformed = N(
                    L.CausalAtrousConv1D(32, 2, border_mode='same', atrous_rate=dilation)(h_down)
                )
                h_up = N(
                    L.Conv1D(64, 1, border_mode='same')(transformed),
                )
                h = L.merge([h, h_up], mode='sum')
            h = N(
                L.Conv1D(64, 1, border_mode='same')(h),
            )
            h = L.Conv1D(action_flat_dim, 1, border_mode='same')(h)
            return h

        decoder_output = _new_decoder_output()

        model = Model(input=[policy_obs, demo_obs], output=decoder_output)

        self.model = model
        self.demo_obs_pool = None
        # demo embedding can be precomputed
        self.demo_obs = None

        obs_var = self.observation_space.new_tensor_variable(extra_dims=2, name="obs")
        demo_obs_var = self.new_demo_vars()["demo_obs"]
        self._f_dist_info = tensor_utils.compile_function(
            [obs_var, demo_obs_var],
            self.dist_info_sym(obs_var, dict(demo_obs=demo_obs_var)),
            extra_feed={K.learning_phase(): 0}
        )

        KerasPowered.__init__(self, [model])

    def new_demo_vars(self):
        return dict(
            demo_obs=self.observation_space.new_tensor_variable(extra_dims=2, name="demo_obs")
        )

    def process_demo_data(self, paths):
        demo_obs = np.asarray([p["observations"] for p in paths])
        return dict(demo_obs=demo_obs)

    def dist_info_sym(self, obs_var, demo_vars):
        demo_obs_var = demo_vars["demo_obs"]
        nn_output = self.model([obs_var, demo_obs_var])
        nn_output = tf.reshape(nn_output, (-1, space_to_dist_dim(self.action_space)))
        return output_to_info(nn_output, self.action_space)

    @cached_property
    def distribution(self):
        return space_to_distribution(self.action_space)

    def inform_task(self, task_id, env, paths, obs):
        self.demo_obs_pool = obs

    def reset(self, dones=None):
        if dones is None:
            dones = [True]
        dones = np.cast['bool'](dones)
        cnt = int(np.sum(dones))
        if cnt > 0:
            demo_ids = np.random.choice(len(self.demo_obs_pool), size=cnt, replace=True)
            # only take the last time step
            demo_obs = self.demo_obs_pool[demo_ids, -1]
            if self.demo_obs is None or len(self.demo_obs) != len(dones):
                self.demo_obs = demo_obs
            else:
                self.demo_obs[dones] = demo_obs

    def get_action(self, observation):
        actions, outputs = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in outputs.items()}

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        agent_infos = self._f_dist_info(flat_obs, self.demo_obs)
        actions = self.distribution.sample(agent_infos)
        return actions, agent_infos
