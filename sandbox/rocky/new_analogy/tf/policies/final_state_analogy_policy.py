from cached_property import cached_property
from rllab.core.serializable import Serializable
import numpy as np

import keras.layers as L
from keras.models import Model

from sandbox.rocky.tf.core.keras_powered import KerasPowered
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.regressors.auto_mlp_regressor import space_to_dist_dim, output_to_info, space_to_distribution
import tensorflow as tf
import keras.backend as K
import keras.backend.tensorflow_backend as KK
KK.conv2d


class FinalStateAnalogyPolicy(KerasPowered, Serializable):
    def __init__(
            self,
            env_spec,
            hidden_sizes=(128, 128),
            embedding_hidden_sizes=(),
            activation='relu',
            batch_norm=False,
    ):
        Serializable.quick_init(self, locals())
        obs_dim = env_spec.observation_space.flat_dim
        action_flat_dim = space_to_dist_dim(env_spec.action_space)
        self.env_spec = env_spec
        self.observation_space = env_spec.observation_space
        self.action_space = env_spec.action_space

        demo_obs = L.Input(shape=(obs_dim,))
        enc = demo_obs
        for hidden_size in embedding_hidden_sizes:
            enc = L.Dense(hidden_size)(enc)
            if batch_norm:
                enc = L.BatchNormalization()(enc)
            enc = L.Activation(activation)(enc)

        obs = L.Input(shape=(obs_dim,))
        dec = L.merge([obs, enc], mode='concat')
        for hidden_size in hidden_sizes:
            dec = L.Dense(hidden_size)(dec)
            if batch_norm:
                dec = L.BatchNormalization()(dec)
            dec = L.Activation(activation)(dec)
        dec = L.Dense(action_flat_dim)(dec)

        model = Model(input=[obs, demo_obs], output=dec)

        self.model = model
        self.demo_obs_pool = None
        self.demo_obs = None

        obs_var = self.observation_space.new_tensor_variable(extra_dims=1, name="obs")
        demo_obs_var = self.new_demo_vars()["demo_obs"]
        self._f_dist_info = tensor_utils.compile_function(
            [obs_var, demo_obs_var],
            self.dist_info_sym(tf.expand_dims(obs_var, 1), dict(demo_obs=demo_obs_var)),
            extra_feed={K.learning_phase(): 0}
        )

        KerasPowered.__init__(self, [model])

    def new_demo_vars(self):
        return dict(
            demo_obs=self.observation_space.new_tensor_variable(extra_dims=1, name="demo_obs")
        )

    def process_demo_data(self, paths):
        demo_obs = np.asarray([p["observations"][-1] for p in paths])
        return dict(demo_obs=demo_obs)

    def dist_info_sym(self, obs_var, demo_vars):
        demo_obs = demo_vars["demo_obs"]
        per_demo_batch_size = tf.shape(obs_var)[1]
        demo_obs = tf.tile(demo_obs, tf.pack([per_demo_batch_size, 1]))
        obs_var = tf.reshape(obs_var, (-1, self.observation_space.flat_dim))
        nn_output = self.model([obs_var, demo_obs])
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
