import numpy as np
import cgtcompat as theano
import cgtcompat.tensor as T
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import operator

def unflatten_tensors(flattened, tensor_shapes):
    tensor_sizes = map(lambda shape: reduce(operator.mul, shape), tensor_shapes)
    indices = np.cumsum(tensor_sizes)[:-1]
    return map(lambda pair: np.reshape(pair[0], pair[1]), zip(np.split(flattened, indices), tensor_shapes))


class ParamLayer(L.Layer):
    def __init__(self, incoming, num_units, param=lasagne.init.Constant(0.), trainable=True, **kwargs):
        super(ParamLayer, self).__init__(incoming, **kwargs)
        self.num_units = num_units
        self.param = self.add_param(param, (1, num_units), name="param", trainable=trainable)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        return T.tile(self.param, (input.shape[0], 1))

class SimpleNNPolicy(object):

    def __init__(self, hidden_sizes=[32,32], nonlinearity=NL.tanh):
        self.input_var = T.matrix('input')
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        self.observation_shape = (17,)#mdp.observation_shape
        self.n_actions = 3#mdp.action_dims

        mean_layer, log_std_layer = self.new_network_outputs(
            self.observation_shape,
            self.n_actions,
            self.input_var
            )

        mean_var = L.get_output(mean_layer)
        log_std_var = L.get_output(log_std_layer)
        self.pdep_vars = [mean_var, log_std_var]

        self.params = L.get_all_params(
            L.concat([mean_layer, log_std_layer]),
            trainable=True
        )
        self.param_shapes = map(
            lambda x: theano.compat.get_value(x, borrow=True).shape,
            self.params
        )
        self.param_dtypes = map(
            lambda x: theano.compat.get_value(x, borrow=True).dtype,
            self.params
        )


    def likelihood_ratio(self, old_pdep_vars, new_pdep_vars, action_var):
        old_mean, old_log_std = old_pdep_vars
        new_mean, new_log_std = new_pdep_vars
        logli_new = log_normal_pdf(action_var, new_mean, new_log_std)
        logli_old = log_normal_pdf(action_var, old_mean, old_log_std)
        return T.exp(T.sum(logli_new - logli_old, axis=1))

    def new_network_outputs(self, observation_shape, n_actions, input_var):
        l_input = L.InputLayer(shape=(None, observation_shape[0]), input_var=input_var)
        l_hidden = l_input
        for idx, hidden_size in enumerate(self.hidden_sizes):
            l_hidden = L.DenseLayer(l_hidden, num_units=hidden_size, nonlinearity=self.nonlinearity, W=lasagne.init.Normal(0.1), name="h%d" % idx)
        mean_layer = L.DenseLayer(l_hidden, num_units=n_actions, nonlinearity=None, W=lasagne.init.Normal(0.01), name="output_mean")
        log_std_layer = ParamLayer(l_input, num_units=n_actions, param=lasagne.init.Constant(0.), name="output_log_std")
        self.log_std_var = L.get_all_params(log_std_layer, trainable=True)[0]
        return mean_layer, log_std_layer

    def unflatten_tensors(flattened, tensor_shapes):
        tensor_sizes = map(lambda shape: reduce(operator.mul, shape), tensor_shapes)
        indices = np.cumsum(tensor_sizes)[:-1]
        return map(lambda pair: np.reshape(pair[0], pair[1]), zip(np.split(flattened, indices), tensor_shapes))

    def set_param_values(self, flattened_params):
        param_values = unflatten_tensors(flattened_params, self.param_shapes)
        for param, dtype, value in zip(
                self.params,
                self.param_dtypes,
                param_values
                ):
            theano.compat.set_value(param, value.astype(dtype))


def log_normal_pdf(x, mean, log_std):
    normalized = (x - mean) / T.exp(log_std)
    return -0.5*T.square(normalized) - np.log((2*np.pi)**0.5) - log_std

def new_surrogate_loss(policy, input_var, Q_est_var, old_pdep_vars, action_var):
    pdep_vars = policy.pdep_vars
    lr = policy.likelihood_ratio(old_pdep_vars, pdep_vars, action_var)
    # KL divergence ignored for this test
    return -T.mean(lr * Q_est_var)

#mdp = HopperMDP()
policy = SimpleNNPolicy()
input_var = policy.input_var
Q_est_var = T.vector('Q_est')
old_pdep_vars = [T.matrix('old_pdep_%d' % i) for i in range(len(policy.pdep_vars))]
action_var = T.matrix('action')

loss = new_surrogate_loss(policy, input_var, Q_est_var, old_pdep_vars, action_var)
compute_loss = theano.function([input_var, Q_est_var] + old_pdep_vars + [action_var], loss)


data = np.load('check.npz')
policy.set_param_values(data['cur_policy_params'])
print compute_loss(data['all_obs'], data['Q_est'], data['pdep_0'], data['pdep_1'], data['actions'])
