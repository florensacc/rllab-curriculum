from __future__ import print_function
from __future__ import absolute_import
from rllab.policies.base import StochasticPolicy
import numpy as np
from rllab.core.network import ConvNetwork, MLP
from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.misc import ext
from rllab.misc import special
from rllab.distributions.recurrent_categorical import RecurrentCategorical
import theano.tensor as TT
import theano
import lasagne
from theano.tensor.opt import register_canonicalize
# import tensorflow as tf


import lasagne.layers as L
import lasagne.nonlinearities as NL

floatX = theano.config.floatX


def time_shift(mat):
    m, n = mat.shape
    return np.concatenate([mat[:, 1:], np.zeros((m, 1))], axis=-1)


class CustomGrad(theano.compile.ViewOp):
    def make_node(self, x, known):
        return theano.gof.Apply(self, [x, known], [x.type()])

    def perform(self, node, inp, out):
        x, _ = inp
        z, = out
        z[0] = x

    def c_code(self, node, nodename, inp, out, sub):
        # import ipdb; ipdb.set_trace()
        iname, _ = inp
        oname, = out
        fail = sub['fail']

        itype = node.inputs[0].type.__class__
        if itype in self.c_code_and_version:
            code, version = self.c_code_and_version[itype]
            return code % locals()

        # Else, no C code
        return super(CustomGrad, self).c_code(node, nodename, inp, out, sub)

    def grad(self, args, g_outs):
        return [g_outs[0], g_outs[0]]

    def infer_shape(self, node, shapes):
        return [shapes[0]]


custom_grad = CustomGrad()
register_canonicalize(theano.gof.PatternSub((custom_grad, 'x', 'y'), 'x'), name='remove_custom_grad')


def my_batched_dot(A, B):
    """Batched version of dot-product.

    For A[dim_1, dim_2, dim_3] and B[dim_1, dim_3, dim_4] this
    is \approx equal to:

    for i in range(dim_1):
        C[i] = tensor.dot(A[i], B[i])

    Returns
    -------
        C : shape (dim_1 \times dim_2 \times dim_4)
    """
    C = A.dimshuffle([0, 1, 2, 'x']) * B.dimshuffle([0, 'x', 1, 2])
    return C.sum(axis=-2)


class AttnFilterLayer(L.Layer):
    def __init__(self, incoming, K, T, name=None):
        """
        :param incoming: should have shape Nx4
        :param K: size of the filter
        """
        super(AttnFilterLayer, self).__init__(incoming, name)
        self.K = K
        self.T = T

    def get_output_shape_for(self, input_shape):
        # assert len(input_shape) == 3
        N, _ = input_shape
        output_shape = (N, self.K, self.T)
        return output_shape

    def get_output_for(self, input, **kwargs):
        tol = 1e-4

        center_t = input[:, 0]
        log_delta = input[:, 1]
        log_sigma = input[:, 2]
        # log_gamma = input[:, 3]

        delta = TT.exp(log_delta)
        sigma = TT.exp(log_sigma / 2.)
        # gamma = TT.exp(log_gamma).dimshuffle(0, 'x', 'x')

        # normalize coordinates
        center_t = (center_t + 1.) / 2. * self.T
        delta = (self.T - 1) / (max(1, self.K - 1)) * delta

        rng = TT.arange(self.K, dtype=floatX) - self.K / 2. + 0.5  # e.g.  [1.5, -0.5, 0.5, 1.5]

        muT = center_t.dimshuffle([0, 'x']) + delta.dimshuffle([0, 'x']) * rng

        t = TT.arange(self.T, dtype=floatX)

        FT = TT.exp(-(t - muT.dimshuffle([0, 1, 'x'])) ** 2 / 2. / sigma.dimshuffle([0, 'x', 'x']) ** 2)
        FT = FT / (FT.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)

        return FT


class AttnReadLayer(L.MergeLayer):
    def __init__(self, l_in, l_filter, l_attn_param, name=None):
        super(AttnReadLayer, self).__init__([l_in, l_filter, l_attn_param], name=name)
        self.l_in = l_in
        self.l_filter = l_filter
        self.l_attn_param = l_attn_param

    def get_output_shape_for(self, input_shapes):
        in_shape, filter_shape, attn_param_shape = input_shapes
        N, A, T = in_shape
        _, K, _ = filter_shape
        return (N, A, K)

    def get_output_for(self, inputs, **kwargs):
        in_var, filter_var, attn_param = inputs
        log_gamma = attn_param[:, 3]
        gamma = TT.exp(log_gamma).dimshuffle(0, 'x', 'x')
        read_result = my_batched_dot(in_var, filter_var.transpose([0, 2, 1]))  # this should have size NxAxK
        read_result = gamma * read_result
        return read_result


class AttnWriteLayer(L.MergeLayer):
    def __init__(self, l_in, l_filter, l_attn_param, name=None):
        super(AttnWriteLayer, self).__init__([l_in, l_filter, l_attn_param], name=name)
        self.l_in = l_in
        self.l_filter = l_filter
        self.l_attn_param = l_attn_param

    def get_output_shape_for(self, input_shapes):
        in_shape, filter_shape, attn_param_shape = input_shapes
        N, A, K = in_shape
        _, _, T = filter_shape
        return (N, A, T)

    def get_output_for(self, inputs, **kwargs):
        in_var, filter_var, attn_param = inputs
        log_gamma = attn_param[:, 3]
        gamma = TT.exp(log_gamma).dimshuffle(0, 'x', 'x')
        return 1. / gamma * my_batched_dot(in_var, filter_var)


class UpdateALayer(L.MergeLayer):
    def __init__(self, l_A, l_g, l_write_A, name=None):
        super(UpdateALayer, self).__init__([l_A, l_g, l_write_A], name=name)
        self.l_A = l_A
        self.l_g = l_g
        self.l_write_A = l_write_A

    def get_output_shape_for(self, input_shapes):
        A_shape, g_shape, write_A_shape = input_shapes
        return A_shape
        # import ipdb;
        # ipdb.set_trace()

    def get_output_for(self, inputs, **kwargs):
        A, g, write_A = inputs
        # apply time shift operator to A
        # A should have shape N*A*T
        N, action_dim, _ = A.shape
        shift_A = TT.concatenate([
            A[:, :, 1:],
            TT.zeros((N, action_dim, 1)),
        ], axis=-1)
        # g should have shape N*1
        return shift_A + g.reshape((-1,)).dimshuffle(0, 'x', 'x') * write_A


class UpdateCLayer(L.MergeLayer):
    def __init__(self, l_c, l_g, l_write_c, name=None):
        super(UpdateCLayer, self).__init__([l_c, l_g, l_write_c], name=name)
        self.l_c = l_c
        self.l_g = l_g
        self.l_write_c = l_write_c
        self.b = self.add_param(np.array(0., dtype=floatX), shape=tuple(), name="b")

    def get_output_shape_for(self, input_shapes):
        c_shape, g_shape, write_c_shape = input_shapes
        return c_shape
        # import ipdb;
        # ipdb.set_trace()

    def get_output_for(self, inputs, **kwargs):
        c, g, write_c = inputs
        N = c.shape[0]
        shift_c = TT.concatenate([
            c[:, :, 1:],
            TT.zeros((N, 1, 1)),
        ], axis=-1)
        g = g.reshape((-1,)).dimshuffle(0, 'x', 'x')
        return shift_c * (1 - g) + g * TT.nnet.sigmoid(self.b + write_c)


class STRAWPolicy(StochasticPolicy, LasagnePowered):
    def __init__(self, env_spec):
        self.T = T = 500
        self.K = K = 10
        self.action_dim = action_dim = env_spec.action_space.flat_dim
        self.A = None
        self.c = None

        feature_dim = 128

        feature_network = ConvNetwork(
            name="feature_network",
            input_shape=env_spec.observation_space.shape,
            output_dim=feature_dim,
            conv_filters=(16, 16),
            conv_filter_sizes=(3, 3),
            conv_strides=(1, 1),
            conv_pads=('same', 'same'),
            hidden_sizes=tuple(),  # 10,),#tuple(10),
            hidden_nonlinearity=NL.rectify,
            # TODO not yet sure about this
            output_nonlinearity=NL.tanh,
        )

        l_obs = feature_network.input_layer

        l_feature = feature_network.output_layer

        l_A_in = L.InputLayer(
            shape=(None, action_dim, self.T),
            name="A_input"
        )
        l_c_in = L.InputLayer(
            shape=(None, 1, self.T),
            name="c_input",
        )
        l_g_in = L.InputLayer(
            shape=(None, 1),
            name="g_input",
        )

        l_A_attn_param = L.DenseLayer(
            l_feature,
            nonlinearity=NL.identity,
            num_units=4,
        )

        l_A_attn_filter = AttnFilterLayer(
            l_A_attn_param,
            K=self.K,
            T=self.T,
        )

        # attentively read the current state of the action-plan
        l_beta = AttnReadLayer(
            l_A_in,
            l_A_attn_filter,
            l_A_attn_param
        )

        # compute intermediate representation
        l_xi = L.concat([L.flatten(l_beta), l_feature])

        l_write_A = AttnWriteLayer(
            l_in=L.reshape(
                L.DenseLayer(
                    l_xi,
                    num_units=action_dim * K,
                ),
                ([0], action_dim, K)
            ),
            l_filter=l_A_attn_filter,
            l_attn_param=l_A_attn_param,
        )

        l_c_attn_filter = AttnFilterLayer(
            l_A_attn_param,
            K=1,
            T=self.T,
        )

        l_write_c = AttnWriteLayer(
            l_in=L.reshape(
                ParamLayer(
                    l_g_in, 1, param=lasagne.init.Constant(np.cast[floatX](40)),
                ),
                ([0], 1, 1),
            ),
            l_filter=l_c_attn_filter,
            l_attn_param=l_A_attn_param,
        )

        l_next_A = UpdateALayer(
            l_A_in,
            l_g_in,
            l_write_A,
        )

        l_next_c = UpdateCLayer(
            l_c_in,
            l_g_in,
            l_write_c
        )

        self.l_obs = l_obs
        self.l_A_in = l_A_in
        self.l_c_in = l_c_in
        self.l_g_in = l_g_in
        self.l_next_A = l_next_A
        self.l_next_c = l_next_c

        self.init_A_var = theano.shared(np.zeros((self.action_dim, self.T), dtype=floatX), name="init_A")
        self.init_c_var = theano.shared(np.ones((1, self.T), dtype=floatX), name="init_c")

        next_A_var, next_c_var = L.get_output([l_next_A, l_next_c])
        a_prob_var = TT.nnet.softmax(next_A_var[:, :, 0])
        self.f_new_A_c_prob = ext.compile_function(
            inputs=[l_obs.input_var, l_A_in.input_var, l_c_in.input_var, l_g_in.input_var],
            outputs=[next_A_var, next_c_var, a_prob_var],
            log_name="f_new_A_c"
        )

        super(STRAWPolicy, self).__init__(env_spec)
        LasagnePowered.__init__(self, [l_next_A, l_next_c])

        self.reset()

    def reset(self):
        # The state of the policy consists of an action-plan matrix of size AxT and a commit vector of length T
        self.A = self.init_A_var.get_value()
        # Initialize to always replan - this will be overwritten on the first time step
        self.c = self.init_c_var.get_value()

    def get_action(self, observation):
        # First sample gt~c_1^{t-1}
        g = int(np.random.uniform() < self.c[0, 0])
        flat_obs = self.observation_space.flatten(observation)
        if g == 1:  # update plan
            new_A, new_c, a_prob = [x[0] for x in self.f_new_A_c_prob([flat_obs], [self.A], [self.c], [[g]])]
        else:
            new_A = time_shift(self.A)
            new_c = time_shift(self.c)
            a_prob = special.softmax(new_A[:, 0])
        self.A = new_A
        self.c = new_c
        action = special.weighted_sample(a_prob, np.arange(self.action_dim))
        return action, dict(prob=a_prob, g=np.array([g]))

    @property
    def recurrent(self):
        return True

    @property
    def distribution(self):
        return RecurrentCategorical(self.action_dim)

    @property
    def state_info_keys(self):
        return ["g"]

    def dist_info_sym(self, obs_var, state_info_vars):
        # obs_var: N*T*S
        # g_var: N*T*1
        N = obs_var.shape[0]
        g_var = state_info_vars["g"]

        init_A = TT.tile(self.init_A_var.dimshuffle('x', 0, 1), (N, 1, 1))
        init_c = TT.tile(self.init_c_var.dimshuffle('x', 0, 1), (N, 1, 1))

        def step(obs, g, A, c):
            next_A_var, next_c_var = L.get_output(
                [self.l_next_A, self.l_next_c],
                inputs={
                    self.l_obs: obs,
                    self.l_A_in: A,
                    self.l_c_in: c,
                    self.l_g_in: custom_grad(g, c[:, :, 0])
                }
            )
            return TT.unbroadcast(next_A_var, *range(next_A_var.ndim)), TT.unbroadcast(next_c_var,
                                                                                       *range(next_c_var.ndim))
            # a_prob_var = TT.nnet.softmax(next_A_var[:, :, 0])
            # self.f_new_A_c_prob = ext.compile_function(
            # inputs=[l_obs.input_var, l_A_in.input_var, l_c_in.input_var, l_g_in.input_var],
            # import ipdb; ipdb.set_trace()
            #
            # pass

        (all_A, all_c), _ = theano.scan(
            step,
            sequences=[obs_var.dimshuffle(1, 0, 2), g_var.dimshuffle(1, 0, 2)],
            outputs_info=[init_A, init_c]
        )
        all_A = all_A.dimshuffle(1, 0, 2, 3)
        N, T, A, _ = all_A.shape
        # size: NxTxAxT
        return dict(prob=TT.nnet.softmax(all_A[:, :, :, 0].reshape((N * T, A))).reshape((N, T, A)))
