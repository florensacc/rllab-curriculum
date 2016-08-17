import lasagne.layers as L
import lasagne.nonlinearities as NL
import theano
import theano.tensor as T
from rllab.core.network import MLP
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.distributions.delta import Delta
from rllab.misc import ext
from rllab.policies.base import Policy
import numpy as np

class DeterministicMLPPolicy(Policy,LasagnePowered,Serializable):
    def __init__(
            self,
            env,
            hidden_sizes=[32,32],
            hidden_nonlinearity=NL.tanh,
            bound_output=True,
        ):
        Serializable.quick_init(self, locals())

        # constrain the output actions by tanh and linear rescaling
        if bound_output:
            act_bounds = env.action_space.bounds
            ll = np.array(act_bounds)[0]
            uu = np.array(act_bounds)[1]
            aa = (uu-ll)/2
            bb = (uu+ll)/2
            def output_nonlinearity(x):
                return T.tanh(x) * aa + bb
        else:
            def output_nonlinearity(x):
                return x

        # construct the network
        obs_dim = env.observation_space.flat_dim
        act_dim = env.action_space.flat_dim
        network = MLP(
            input_shape=(obs_dim,),
            output_dim=act_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=output_nonlinearity,
        )
        self.network = network
        l_out = network.output_layer
        act_var = L.get_output(l_out).flatten()
        obs_var = network.input_layer.input_var

        # output
        LasagnePowered.__init__(self, [l_out])
        self._get_action = ext.compile_function(
            inputs=[obs_var],
            outputs=act_var,
        )

        # derivatives
        theta = self.get_params()
        flat_grads = []
        for i in range(act_dim):
            # grad w.r.t. NN params are grouped by layers; so need to flatten
            grad = theano.grad(act_var[i], wrt=theta, disconnected_inputs='warn')
            flat_grad = ext.flatten_tensor_variables(grad)
            flat_grads.append(flat_grad)
        jacobian_theta = T.stack(flat_grads)
        self._pi_theta = ext.compile_function(
            inputs=[obs_var],
            outputs=jacobian_theta,
        )

        flat_grads = []
        for i in range(act_dim):
            # beware of the final ravel()
            grad = theano.grad(act_var[i], wrt=obs_var, disconnected_inputs='warn').ravel()
            flat_grads.append(grad)
        jacobian_s = T.stack(flat_grads)
        self._pi_s = ext.compile_function(
            inputs=[obs_var],
            outputs=jacobian_s,
        )

        self._dist = Delta()

    def get_action(self,state):
        agent_info = dict()
        action = self._get_action([state])
        return action, agent_info

    def get_theta(self):
        return self.get_param_values()

    def set_theta(self,value):
        self.set_param_values(value)

    def pi_theta(self,state):
        return self._pi_theta([state])

    def pi_s(self,state):
        return self._pi_s([state])

    def reset(self):
        pass

    @property
    def distribution(self):
        return self._dist

    def log_diagnostics(self,path):
        pass
