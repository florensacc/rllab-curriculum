import tensorflow as tf

from policies.nn_policy import FeedForwardPolicy
from qfunctions.naf_qfunction import NAFQFunction
from qfunctions.quadratic_qf import QuadraticQF
from rllab.core.serializable import Serializable
from vfunction.mlp_vfunction import MlpStateNetwork


class QuadraticNAF(NAFQFunction):
    def __init__(
            self,
            scope_name,
            env_spec,
            **kwargs
    ):
        Serializable.quick_init(self, locals())
        super(QuadraticNAF, self).__init__(
            scope_name,
            env_spec,
            1,
            **kwargs
        )

    def _create_network(self):
        self.policy = FeedForwardPolicy(
            "mu",
            self.observation_dim,
            self.action_dim,
            observation_hidden_sizes=(200, 200),
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
            )
        # TODO(vpong): fix this hack
        self.observation_input = self.policy.observations_placeholder
        vf_output_dim = 1
        self.vf = MlpStateNetwork(
            "V_function",
            self.env_spec,
            vf_output_dim,
            observation_input=self.observation_input,
            observation_hidden_sizes=(200, 200),
            hidden_W_init=None,
            hidden_b_init=None,
            output_W_init=None,
            output_b_init=None,
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.identity,
        )
        af_output_dim = 1
        self.af = QuadraticQF(
            "advantage_function",
            self.env_spec,
            af_output_dim,
            self.action_input,
            self.observation_input,
            policy=self.policy,
        )
        return self.vf.output + self.af.output

    def get_implicit_policy(self):
        return self.policy

    def get_implicit_value_function(self):
        return self.vf

    def get_implicit_advantage_function(self):
        return self.af

