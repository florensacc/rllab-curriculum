import numpy as np
# from numpy.linalg import inv
import theano
import theano.tensor as TT
from theano.tensor.nlinalg import matrix_inverse as inv

import lasagne.layers as L
import lasagne.nonlinearities as NL

from rllab.spaces import Box
from rllab.core.network import MLP
from rllab.core.serializable import Serializable
from rllab.policies.base import Policy
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.misc import ext

import common.util as util

def lqr(A, B, Q, R, k):
    def recurrence(P):
        tmp1 = TT.dot(A.T, TT.dot(P, A))
        tmp2 = TT.dot(A.T, TT.dot(P, B))
        tmp3 = R + TT.dot(B.T, TT.dot(P, B))
        tmp4 = TT.dot(B.T, TT.dot(P, A))
        return Q + tmp1 - TT.dot(tmp2, TT.dot(inv(tmp3), tmp4))

    Ps, _ = theano.scan(fn=recurrence,
                       outputs_info=[Q],
                       n_steps=k)
    P = Ps[-1]

    return P, TT.dot(inv(R + TT.dot(B.T, TT.dot(P, B))), TT.dot(B.T, TT.dot(P, A)))

class RecurrentLQRPolicy(Policy, Serializable):
    def __init__(self, env_spec, Q, R, state_dim=None, physical_dim=None, recurrences=10):
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.observation_space, Box)
        assert isinstance(env_spec.action_space, Box)
        assert state_dim or physical_dim

        observation_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim

        if state_dim and physical_dim:
            assert observation_dim == state_dim + physical_dim
        elif physical_dim:
            state_dim = observation_dim - physical_dim
        else:
            physical_dim = observation_dim - state_dim

        assert R.shape == (action_dim, action_dim)

        float_t = theano.config.floatX
        obs_var = env_spec.observation_space.new_tensor_variable('obs', 1)

        # self._A = util.init_weights(observation_dim, observation_dim)
        # self._B = util.init_weights(observation_dim, action_dim)
        self._dynamics_net = MLP(
                output_dim=state_dim * (state_dim + action_dim),
                hidden_sizes=[50, 100],
                hidden_nonlinearity=NL.rectify,
                output_nonlinearity=NL.rectify,
                # input_var=obs_var[-physical_dim:],
                input_shape=(physical_dim,)
        )
        self._Q = theano.shared(Q.astype(float_t))
        self._R = theano.shared(R.astype(float_t))
        self._state_dim = state_dim
        self._physical_dim = physical_dim
        self._action_dim = action_dim
        self._recurrences = recurrences

        action_var = self.get_action_sym(obs_var)
        self._f_actions = ext.compile_function([obs_var], action_var)

        super(RecurrentLQRPolicy, self).__init__(env_spec)

    @property
    def recurrent(self):
        return True

    def get_params_internal(self, **tags):
        # return [self._A, self._B]
        return self._dynamics_net.get_params_internal()

    def get_action(self, observation):
        action = self._f_actions([observation])[0]
        return action, dict()

    def get_actions(self, observations):
        return self._f_actions(observations), dict()

    def get_action_sym(self, obs_var):
        return self._forward(obs_var)

    def _map_dynamics(self, obs_var):
        physical_var = obs_var[:,self._state_dim:]
        net_out = L.get_output(self._dynamics_net.output_layer, inputs=physical_var)[0]
        # With regards to taking only the first row of output:                       ^
        # It's ok because the physical parameters will not change mid-episode
        
        A = net_out[:self._state_dim**2].reshape((self._state_dim, self._state_dim))
        B = net_out[self._state_dim**2:].reshape((self._state_dim, self._action_dim))
        return A, B

    def _forward(self, obs_var):
        # u = TT.dot(W, x)
        # for i in range(k):
        #     tmp1 = TT.dot(B.T, TT.dot(P, TT.dot(B, u)))
        #     tmp2 = TT.dot(B.T, TT.dot(P, TT.dot(A, x)))
        #     u = -TT.dot(inv(R), tmp1+tmp2)

        self._A, self._B = self._map_dynamics(obs_var)
        self._P, self._K = lqr(self._A, self._B, self._Q, self._R, self._recurrences)
        state_var = obs_var[:,:self._state_dim]
        return -TT.dot(self._K, state_var.T).T
