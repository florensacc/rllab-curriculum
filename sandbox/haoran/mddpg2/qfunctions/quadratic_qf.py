import tensorflow as tf
from core import tf_util
from predictors.state_action_network import StateActionNetwork
from vfunction.mlp_vfunction import MlpStateNetwork
from rllab.core.serializable import Serializable


class QuadraticQF(StateActionNetwork):
    def __init__(
            self,
            scope_name,
            env_spec,
            output_dim,
            action_input,
            observation_input,
            policy,
            reuse=False,
    ):
        Serializable.quick_init(self, locals())
        self.policy = policy
        super(QuadraticQF, self).__init__(
            scope_name,
            env_spec,
            output_dim,
            action_input,
            observation_input,
            policy,
        )

    def _create_network(self):
        L_params = MlpStateNetwork(
            "L",
            self.env_spec,
            self.action_dim * self.action_dim,
            observation_input=self.observation_input,
            observation_hidden_sizes=(200, 200),
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.identity,
            )
        L = tf_util.vec2lower_triangle(L_params.output, self.action_dim)

        # P_matrix = tf.matmul(
        #     L_lower_triangle_matrix,
        #     L_lower_triangle_matrix,
        #     transpose_b=True,
        # )
        delta = self.action_input - self.policy.output
        h1 = tf.expand_dims(delta, 1)
        h1 = tf.batch_matmul(h1, L)  # batch:1:dimA
        h1 = tf.squeeze(h1, [1])  # batch:dimA
        return -tf.constant(0.5) * tf.reduce_sum(h1 * h1, 1)  # batch
