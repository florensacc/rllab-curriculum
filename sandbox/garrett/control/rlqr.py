import numpy as np
# from numpy.linalg import inv
import theano
import theano.tensor as TT
from theano.tensor.nlinalg import matrix_inverse as inv

from rllab.spaces import Box
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
    def __init__(self, env_spec, Q, R, k=10):
        Serializable.quick_init(self, locals())
        assert isinstance(env_spec.observation_space, Box)
        assert isinstance(env_spec.action_space, Box)

        observation_dim = env_spec.observation_space.flat_dim
        action_dim = env_spec.action_space.flat_dim
        assert R.shape == (action_dim, action_dim)

        float_t = theano.config.floatX

        self._A = util.init_weights(observation_dim, observation_dim)
        self._B = util.init_weights(observation_dim, action_dim)
        self._Q = theano.shared(Q.astype(float_t))
        self._R = theano.shared(R.astype(float_t))
        self._k = k
        self._P, self._K = lqr(self._A, self._B, self._Q, self._R, self._k)


        obs_var = env_spec.observation_space.new_tensor_variable('obs', 1)
        action_var = self.get_action_sym(obs_var)
        self._f_actions = ext.compile_function([obs_var], action_var)

        super(RecurrentLQRPolicy, self).__init__(env_spec)

    @property
    def recurrent(self):
        return True

    def get_params_internal(self, **tags):
        return [self._A, self._B]

    def get_action(self, observation):
        action = self._f_actions([observation])[0]
        return action, dict()

    def get_actions(self, observations):
        return self._f_actions(observations), dict()

    def get_action_sym(self, obs_var):
        return self._forward(obs_var)

    def _forward(self, obs_var):
        # u = TT.dot(W, x)
        # for i in range(k):
        #     tmp1 = TT.dot(B.T, TT.dot(P, TT.dot(B, u)))
        #     tmp2 = TT.dot(B.T, TT.dot(P, TT.dot(A, x)))
        #     u = -TT.dot(inv(R), tmp1+tmp2)

        return -TT.dot(self._K, obs_var.T).T
