


from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L
import tensorflow as tf
from rllab.core.serializable import Serializable
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.misc import tensor_utils
import numpy as np


class DiscreteConvQFunction(LayersPowered, Serializable):
    def __init__(
            self,
            env_spec,
            name,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.relu,
    ):
        Serializable.quick_init(self, locals())

        with tf.variable_scope(name):
            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim

            l_obs = L.InputLayer(
                shape=(None, 84*84),
                name="input",
            )
            l_hid = L.reshape(
                l_obs,
                [-1, 84, 84, 1],
                name="rs"
            )
            l_hid = L.Conv2DLayer(
                l_hid,
                "conv1",
                num_filters=32,
                filter_size=8,
                stride=4,
            )
            l_hid = L.Conv2DLayer(
                l_hid,
                "conv2",
                num_filters=64,
                filter_size=4,
                stride=2,
            )
            l_hid = L.Conv2DLayer(
                l_hid,
                "conv3",
                num_filters=64,
                filter_size=3,
                stride=1,
            )
            l_hid = L.DenseLayer(
                l_hid,
                "h1",
                512,
                nonlinearity=tf.nn.relu,
            )
            l_q = L.DenseLayer(
                l_hid,
                "out",
                action_dim
            )

            self.l_obs = l_obs
            self.l_q = l_q
            self.observation_space = env_spec.observation_space
            self.action_dim = action_dim

            self.f_qval = tensor_utils.compile_function(
                [l_obs.input_var],
                self.full_qval_sym(l_obs.input_var)
            )

            LayersPowered.__init__(self, [l_q])

    def full_qval_sym(self, obs_var):
        return L.get_output(self.l_q, {self.l_obs: obs_var})

    def qval_sym(self, obs_var, action_var):
        qvals = self.full_qval_sym(obs_var)
        action_var = tf.cast(action_var, tf.float32)
        return tf.reduce_sum(qvals * action_var, -1)

    def argmax_qval_sym(self, obs_var):
        qvals = self.full_qval_sym(obs_var)
        return tf.nn.embedding_lookup(
            np.eye(self.action_dim, dtype=np.float32),
            tf.argmax(qvals, dimension=1)
        )

    def reset(self, dones=None):
        pass

    def get_action(self, observation):
        actions, infos = self.get_actions([observation])
        return actions[0], {k: v[0] for k, v in infos.items()}

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        qvals = self.f_qval(flat_obs)
        actions = np.argmax(qvals, axis=1)
        return actions, dict()

