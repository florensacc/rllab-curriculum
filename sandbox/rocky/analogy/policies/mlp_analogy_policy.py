from sandbox.rocky.analogy.policies.base import AnalogyPolicy
import tensorflow as tf
from sandbox.rocky.tf.core.network import MLP
from sandbox.rocky.tf.core.layers_powered import LayersPowered
import sandbox.rocky.tf.core.layers as L


class MLPAnalogyPolicy(AnalogyPolicy, LayersPowered):
    def __init__(self, env_spec):
        self.obs_dim = env_spec.observation_space.flat_dim
        self.action_dim = env_spec.action_space.flat_dim
        self.action_network = MLP(
            input_shape=(self.obs_dim,),
            output_dim=self.action_dim,
            output_nonlinearity=None,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.relu,
            name="action_network",
        )

        self.l_out = self.action_network.output_layer

        AnalogyPolicy.__init__(self, env_spec=env_spec)
        LayersPowered.__init__(self, self.l_out)

        self.obs_var = env_spec.observation_space.new_tensor_variable(name="obs", extra_dims=2)
        self.action_var = self.action_sym(self.obs_var)

    def action_sym(self, obs_var, **kwargs):
        return tf.reshape(
            L.get_output(
                self.action_network.output_layer,
                tf.reshape(obs_var, [-1, self.obs_dim])
            ), tf.pack([tf.shape(obs_var)[0], -1, self.action_dim])
        )

    def reset(self, dones=None):
        pass

    def get_action(self, obs):
        flat_obs = self.observation_space.flatten(obs)
        sess = tf.get_default_session()
        action = sess.run(self.action_var, feed_dict={self.obs_var: [[flat_obs]]})[0, 0]
        return action

    def apply_demo(self, path):
        pass
