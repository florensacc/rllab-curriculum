from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import tensorflow as tf
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.misc import special
import sandbox.rocky.tf.core.layers as L
from sandbox.rocky.tf.policies.base import StochasticPolicy
from sandbox.rocky.tf.misc import tensor_utils
from sandbox.rocky.tf.core.network import ConvNetwork
from sandbox.rocky.tf.distributions.recurrent_categorical import RecurrentCategorical
from sandbox.rocky.tf.core.layers_powered import LayersPowered


def time_shift(mat):
    m, n = mat.shape
    return np.concatenate([mat[:, 1:], np.zeros((m, 1))], axis=-1)


custom_py_cnt = 0


def custom_grad(x, y):
    global custom_py_cnt
    custom_py_cnt += 1
    func_name = "CustomPyFunc%d" % custom_py_cnt

    def _func(x, y):
        return x

    @tf.RegisterGradient(func_name)
    def _grad(op, grad):
        return grad, grad

    @tf.RegisterShape(func_name)
    def _shape(op):
        return [op.inputs[0].get_shape(), op.inputs[0].get_shape()]

    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": func_name}):
        out = tf.py_func(_func, [x, y], [x.dtype])[0]
        out.set_shape(x.get_shape())
        return out


class AttnFilterLayer(L.Layer):
    def __init__(self, incoming, K, T, name):
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

        delta = tf.exp(log_delta)
        sigma = tf.exp(log_sigma / 2.)

        # normalize coordinates
        center_t = (center_t + 1.) / 2. * self.T
        delta = (self.T - 1) / (max(1, self.K - 1)) * delta

        rng = tf.cast(tf.range(self.K), tf.float32) - self.K / 2. + 0.5  # e.g.  [1.5, -0.5, 0.5, 1.5]

        muT = tf.expand_dims(center_t, 1) + tf.expand_dims(delta, 1) * rng

        t = tf.cast(tf.range(self.T), tf.float32)

        FT = tf.exp(-(t - tf.expand_dims(muT, 2)) ** 2 / 2. / tf.expand_dims(tf.expand_dims(sigma, 1), 2) ** 2)
        FT = FT / (tf.reduce_sum(FT, reduction_indices=2, keep_dims=True) + tol)

        return FT


class AttnReadLayer(L.MergeLayer):
    def __init__(self, l_in, l_filter, l_attn_param, name):
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
        gamma = tf.expand_dims(tf.expand_dims(tf.exp(log_gamma), 1), 2)
        read_result = tf.batch_matmul(in_var, tf.transpose(filter_var, [0, 2, 1]))  # this should have size
        # NxAxK
        read_result = gamma * read_result
        return read_result


class AttnWriteLayer(L.MergeLayer):
    def __init__(self, l_in, l_filter, l_attn_param, name):
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
        gamma = tf.expand_dims(tf.expand_dims(tf.exp(log_gamma), 1), 2)
        return 1. / gamma * tf.batch_matmul(in_var, filter_var)


class UpdateALayer(L.MergeLayer):
    def __init__(self, l_A, l_g, l_write_A, name):
        super(UpdateALayer, self).__init__([l_A, l_g, l_write_A], name=name)
        self.l_A = l_A
        self.l_g = l_g
        self.l_write_A = l_write_A

    def get_output_shape_for(self, input_shapes):
        A_shape, g_shape, write_A_shape = input_shapes
        return A_shape

    def get_output_for(self, inputs, **kwargs):
        A, g, write_A = inputs
        # apply time shift operator to A
        # A should have shape N*A*T
        A_shape = tf.shape(A)
        N = A_shape[0]
        action_dim = A_shape[1]
        # N, action_dim, _ = tf.shape(A)
        shift_A = tf.concat(2, [
            A[:, :, 1:],
            tf.zeros(tf.pack([N, action_dim, 1]))
        ])
        # g should have shape N*1
        return shift_A + tf.expand_dims(g, 2) * write_A


class UpdateCLayer(L.MergeLayer):
    def __init__(self, l_c, l_g, l_write_c, name):
        super(UpdateCLayer, self).__init__([l_c, l_g, l_write_c], name=name)
        self.l_c = l_c
        self.l_g = l_g
        self.l_write_c = l_write_c
        self.b = self.add_param(tf.zeros_initializer, shape=tuple(), name="b")

    def get_output_shape_for(self, input_shapes):
        c_shape, g_shape, write_c_shape = input_shapes
        return c_shape

    def get_output_for(self, inputs, **kwargs):
        c, g, write_c = inputs
        N = tf.shape(c)[0]
        shift_c = tf.concat(2, [
            c[:, :, 1:],
            tf.zeros(tf.pack([N, 1, 1])),
        ])
        g = tf.expand_dims(g, 2)
        return shift_c * (1 - g) + g * tf.nn.sigmoid(self.b + write_c)


class STRAWPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            planning_horizon=20,
            patch_horizon=10,
            feature_dim=128,
            sample_decision=True,
            feature_network_cls=None,
            feature_network_args=None):
        Serializable.quick_init(self, locals())
        self.T = T = planning_horizon
        self.K = K = patch_horizon
        self.feature_dim = feature_dim
        self.action_dim = action_dim = env_spec.action_space.flat_dim
        self.sample_decision = sample_decision
        self.A = None
        self.c = None

        with tf.variable_scope(name):
            if feature_network_cls is None:
                feature_network_cls = ConvNetwork
            if feature_network_args is None:
                feature_network_args = dict(
                    name="feature_network",
                    input_shape=env_spec.observation_space.shape,
                    output_dim=feature_dim,
                    conv_filters=(16, 16),
                    conv_filter_sizes=(3, 3),
                    conv_strides=(1, 1),
                    conv_pads=('SAME', 'SAME'),
                    hidden_sizes=tuple(),
                    hidden_nonlinearity=tf.nn.tanh,
                    # TODO not yet sure about this
                    output_nonlinearity=tf.nn.tanh,
                )
            feature_network = feature_network_cls(**feature_network_args)

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
                nonlinearity=tf.identity,
                num_units=4,
                name="A_attn_param",
            )

            l_A_attn_filter = AttnFilterLayer(
                l_A_attn_param,
                K=self.K,
                T=self.T,
                name="A_attn_filter",
            )

            # attentively read the current state of the action-plan
            l_beta = AttnReadLayer(
                l_A_in,
                l_A_attn_filter,
                l_A_attn_param,
                name="read_A"
            )

            # compute intermediate representation
            l_xi = L.concat([L.flatten(l_beta, name="flat_beta"), l_feature], name="xi")

            l_write_A = AttnWriteLayer(
                l_in=L.reshape(
                    L.DenseLayer(
                        l_xi,
                        num_units=action_dim * K,
                        name="project_xi"
                    ),
                    ([0], action_dim, K),
                    name="reshape_project_xi"
                ),
                l_filter=l_A_attn_filter,
                l_attn_param=l_A_attn_param,
                name="write_A"
            )

            l_c_attn_filter = AttnFilterLayer(
                l_A_attn_param,
                K=1,
                T=self.T,
                name="c_attn_filter"
            )

            l_write_c = AttnWriteLayer(
                l_in=L.reshape(
                    L.ParamLayer(
                        l_g_in,
                        name="c_patch",
                        num_units=1,
                        param=tf.constant_initializer(40, dtype=tf.float32),
                    ),
                    ([0], 1, 1),
                    name="reshape_c_patch",
                ),
                l_filter=l_c_attn_filter,
                l_attn_param=l_A_attn_param,
                name="write_c"
            )

            l_next_A = UpdateALayer(
                l_A_in,
                l_g_in,
                l_write_A,
                name="update_A"
            )

            l_next_c = UpdateCLayer(
                l_c_in,
                l_g_in,
                l_write_c,
                name="update_c"
            )

            self.l_obs = l_obs
            self.l_A_in = l_A_in
            self.l_c_in = l_c_in
            self.l_g_in = l_g_in
            self.l_next_A = l_next_A
            self.l_next_c = l_next_c

            self.init_A_var = tf.Variable(np.zeros((self.action_dim, self.T), dtype=np.float32), name="init_A")
            self.init_c_var = tf.Variable(np.ones((1, self.T), dtype=np.float32), name="init_c")

            next_A_var, next_c_var = L.get_output([l_next_A, l_next_c])
            a_prob_var = tf.nn.softmax(next_A_var[:, :, 0])
            self.f_new_A_c_prob = tensor_utils.compile_function(
                inputs=[l_obs.input_var, l_A_in.input_var, l_c_in.input_var, l_g_in.input_var],
                outputs=[next_A_var, next_c_var, a_prob_var],
                log_name="f_new_A_c"
            )

            super(STRAWPolicy, self).__init__(env_spec)
            LayersPowered.__init__(self, [l_next_A, l_next_c])

            # self.reset()

    def reset(self):
        # The state of the policy consists of an action-plan matrix of size AxT and a commit vector of length T
        sess = tf.get_default_session()
        self.A, self.c = sess.run([self.init_A_var, self.init_c_var])

    def get_action(self, observation):
        # First sample gt~c_1^{t-1}
        flat_obs = self.observation_space.flatten(observation)
        if self.sample_decision:
            g = int(np.random.uniform() < self.c[0, 0])
        else:
            g = self.c[0, 0]
        # if g == 1:  # update plan
        new_A, new_c, a_prob = [x[0] for x in self.f_new_A_c_prob([flat_obs], [self.A], [self.c], [[g]])]
        # else:
        #     new_A = time_shift(self.A)
        #     new_c = time_shift(self.c)
        #     a_prob = special.softmax(new_A[:, 0])
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
    def state_info_specs(self):
        return [("g", (1,))]

    def dist_info_sym(self, obs_var, state_info_vars):
        # obs_var: N*T*S
        # g_var: N*T*1
        obs_var = tf.cast(obs_var, tf.float32)

        N = tf.shape(obs_var)[0]
        g_var = state_info_vars["g"]

        # TxAxT
        init_A = tf.tile(
            tf.expand_dims(self.init_A_var, 0),
            tf.pack([N, 1, 1]),
        )
        # Tx1xT
        init_c = tf.tile(
            tf.expand_dims(self.init_c_var, 0),
            tf.pack([N, 1, 1]),
        )

        def step(prev, data):
            A = prev[:, :self.action_dim, :]
            c = prev[:, self.action_dim:, :]
            obs_dim = self.observation_space.flat_dim
            obs = data[:, :obs_dim]
            g = data[:, obs_dim:]
            next_A_var, next_c_var = L.get_output(
                [self.l_next_A, self.l_next_c],
                inputs={
                    self.l_obs: obs,
                    self.l_A_in: A,
                    self.l_c_in: c,
                    self.l_g_in: custom_grad(g, c[:, :, 0]) if self.sample_decision else c[:, :, 0]
                }
            )
            return tf.concat(1, [next_A_var, next_c_var])

        inputs_packed = tf.concat(2, [obs_var, g_var])
        init_packed = tf.concat(1, [init_A, init_c])

        result_packed = tf.scan(
            step,
            elems=tf.transpose(inputs_packed, perm=[1, 0, 2]),
            initializer=init_packed,
        )
        all_A = tf.transpose(result_packed, perm=[1, 0, 2, 3])[:, :, :self.action_dim, :]
        all_A_shape = tf.shape(all_A)
        N = all_A_shape[0]
        T = all_A_shape[1]
        A = all_A_shape[2]
        # size: NxTxAxT
        return dict(prob=tf.reshape(
            tf.nn.softmax(
                tf.reshape(all_A[:, :, :, 0], tf.pack((N * T, A)))
            ),
            tf.pack((N, T, A))
        ))
