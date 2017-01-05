from sandbox.rocky.tf.core.layers_powered import LayersPowered
from rllab.envs.base import EnvSpec
from sandbox.rocky.tf.core.network import MLP
import sandbox.rocky.tf.core.layers as L
import tensorflow as tf

from sandbox.rocky.tf.misc import tensor_utils


class MLPPolicy(LayersPowered):

    def __init__(
            self,
            env_spec: EnvSpec,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=None,
    ):

        self.observation_space = observation_space = env_spec.observation_space
        self.action_space = action_space = env_spec.action_space

        self.network = MLP(
            input_shape=(observation_space.flat_dim,),
            output_dim=action_space.flat_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
            name="action_network"
        )
        self.f_action = tensor_utils.compile_function(
            inputs=[self.network.input_layer.input_var],
            outputs=L.get_output(self.network.output_layer),
        )

    def get_actions(self, observations):
        flat_obs = self.observation_space.flatten_n(observations)
        actions = self.f_action(flat_obs)
        return actions, dict()
