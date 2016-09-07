from __future__ import print_function
import numpy as np
import theano.tensor as T
import lasagne
from lasagne.layers.noise import dropout
from lasagne.layers.normalization import batch_norm

from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.misc import ext
from collections import OrderedDict
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from rllab.misc.special import to_onehot_sym
from sandbox.rein.dynamics_models.utils import enum
from sandbox.rein.dynamics_models.bnn.conv_bnn import BayesianConvLayer, BayesianDeConvLayer, BayesianDenseLayer, \
    BayesianLayer


class DiscreteEmbeddingLayer(lasagne.layers.Layer):
    """
    Discrete embedding layer for counting
    """

    def __init__(self, incoming, num_units, W=lasagne.init.GlorotUniform(),
                 b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify,
                 **kwargs):
        super(DiscreteEmbeddingLayer, self).__init__(incoming, **kwargs)
        self.nonlinearity = (lasagne.nonlinearities.identity if nonlinearity is None
                             else nonlinearity)

        self.num_units = num_units
        num_inputs = int(np.prod(self.input_shape[1:]))

        self.W = self.add_param(W, (num_inputs, num_units), name="W")
        if b is None:
            self.b = None
        else:
            self.b = self.add_param(b, (num_units,), name="b",
                                    regularizable=False)

        self._srng = RandomStreams()

    def get_embedding(self, input):
        return T.cast(T.round(self.get_output_for(input)), 'int32')

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.num_units

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        # Add noise to activation for discretization
        return self.nonlinearity(activation) + self._srng.uniform(low=-0.5, high=0.5)


class IndependentSoftmaxLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_bins, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0), **kwargs):
        super(IndependentSoftmaxLayer, self).__init__(incoming, **kwargs)

        self._num_bins = num_bins
        self.W = self.add_param(W, (self.input_shape[1], self._num_bins), name='W')
        self.b = self.add_param(b, (self._num_bins,), name='b')
        self.pixel_b = self.add_param(
            b,
            (self.input_shape[2], self.input_shape[3], self._num_bins,),
            name='pixel_b'
        )

    def get_output_for(self, input, **kwargs):
        fc = input.dimshuffle(0, 2, 3, 1). \
                 reshape([-1, self.input_shape[1]]). \
                 dot(self.W) + \
             self.b[np.newaxis, :]
        shp = self.get_output_shape_for([-1] + list(self.input_shape[1:]))
        fc_biased = fc.reshape(shp) + self.pixel_b
        out = T.nnet.softmax(
            fc_biased.reshape([-1, self._num_bins])
        )
        return out.reshape(shp)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[2], input_shape[3], self._num_bins


class LogitLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_bins, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0), s=None,
                 **kwargs):
        super(LogitLayer, self).__init__(incoming, **kwargs)

        self._num_bins = num_bins
        self.W = self.add_param(W, (self.input_shape[1], 1), name='W')
        self.pixel_b = self.add_param(
            b,
            (self.input_shape[2], self.input_shape[3], 1),
            name='pixel_b'
        )
        self.s = s

    def get_output_for(self, input, **kwargs):
        def cdf(x, m, s):
            return 1. / (1. + T.exp(- (x - m) / s))

        fc = T.dot(input.dimshuffle(0, 2, 3, 1), self.W) + self.pixel_b
        fc_tiled = T.tile(fc, reps=(1, 1, 1, self._num_bins), ndim=4)
        uniform_bin = (T.arange(self._num_bins) / float(self._num_bins - 1)).dimshuffle('x', 'x', 'x', 0)
        out_a = cdf(uniform_bin + 1 / float(self._num_bins), fc_tiled, self.s)
        out_b = cdf(uniform_bin, fc_tiled, self.s)
        return out_a - out_b + 1e-8

    def get_output_shape_for(self, input_shape):
        return input_shape[0], input_shape[2], input_shape[3], self._num_bins


class OuterProdLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(OuterProdLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return None, input_shapes[1][1], input_shapes[0][1]

    def get_output_for(self, inputs, **kwargs):
        return inputs[0][:, :, np.newaxis] * inputs[1][:, np.newaxis, :]


class HadamardLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(HadamardLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        assert input_shapes[0][1] == input_shapes[1][1]
        return None, input_shapes[0][1]

    def get_output_for(self, inputs, **kwargs):
        return inputs[0] * inputs[1]


class ConcatLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(ConcatLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return None, input_shapes[0][1] + input_shapes[1][1]

    def get_output_for(self, inputs, **kwargs):
        return T.concatenate(inputs, axis=1)


class ConvBNNVIME(LasagnePowered, Serializable):
    """(Convolutional) Bayesian neural network (BNN), according to Blundell2016.

    The input and output to the network is a flat array. Use layers_disc to describe the layers between the input and output
    layers.

    This is a specific implementation of ConvBNN to integrate the action and predict the reward. After the convolutional
    layers, we downscale to a hidden_size, then outer product with the action, upscale to higher layer, split off
    between deconvolution to s' and hidden layer for reward, which predicts the reward.
    """

    # Enums
    OutputType = enum(REGRESSION='regression', CLASSIFICATION='classfication')
    SurpriseType = enum(
        INFGAIN='information gain', COMPR='compression gain', BALD='BALD', VAR='variance', L1='l1')

    def __init__(self,
                 state_dim,
                 action_dim,
                 reward_dim,
                 layers_disc,
                 n_batches,
                 trans_func=lasagne.nonlinearities.rectify,
                 out_func=lasagne.nonlinearities.linear,
                 batch_size=100,
                 n_samples=10,
                 num_train_samples=1,
                 prior_sd=0.5,
                 second_order_update=False,
                 learning_rate=0.0001,
                 surprise_type=SurpriseType.INFGAIN,
                 update_prior=False,
                 update_likelihood_sd=False,
                 use_local_reparametrization_trick=True,
                 likelihood_sd_init=1.0,
                 output_type=OutputType.REGRESSION,
                 num_classes=None,
                 disable_variance=False,  # Disable variances in BNN.
                 debug=False,
                 ind_softmax=False,  # Independent softmax output instead of regression.
                 disable_act_rew_paths=False,  # Disable action and reward modeling, just s -> s' prediction.
                 num_seq_inputs=1,
                 label_smoothing=0,
                 logit_weights=False,
                 logit_output=False
                 ):

        Serializable.quick_init(self, locals())

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim
        self.num_seq_inputs = num_seq_inputs

        self.batch_size = batch_size
        self.transf = trans_func
        self.outf = out_func
        self.n_samples = n_samples
        self.num_train_samples = num_train_samples
        self.prior_sd = prior_sd
        self.layers_disc = layers_disc
        self.n_batches = n_batches
        self.likelihood_sd_init = likelihood_sd_init
        self.second_order_update = second_order_update
        self.learning_rate = learning_rate
        self.surprise_type = surprise_type
        self.update_prior = update_prior
        self.update_likelihood_sd = update_likelihood_sd
        self.use_local_reparametrization_trick = use_local_reparametrization_trick
        self.output_type = output_type
        self.num_classes = num_classes
        self.disable_variance = disable_variance
        self.debug = debug
        self._ind_softmax = ind_softmax
        self._disable_act_rew_paths = disable_act_rew_paths
        self.label_smoothing = label_smoothing
        self._logit_weights = logit_weights
        self._logit_output = logit_output

        assert not (self._logit_output and self._ind_softmax)

        if self._disable_act_rew_paths:
            print('Warning: action and reward paths disabled, do not use a_net or r_net!')

        self.network = None

        if self.output_type == ConvBNNVIME.OutputType.CLASSIFICATION and self.update_likelihood_sd:
            print(
                'Setting output_type=\'classification\' cannot be used with update_likelihood_sd=True, changing to False.')
            self.update_likelihood_sd = False

        if self.disable_variance:
            print('Warning: all noise has been disabled, only using means.')

        # Build network architecture.
        self.build_network()

        # Build model might depend on this.
        LasagnePowered.__init__(self, [self.network])

        # Compile theano functions.
        self.build_model()

        print('num_weights: {}'.format(self.num_weights()))

    def save_params(self):
        layers = filter(lambda l: isinstance(l, BayesianLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        for layer in layers:
            layer.save_params()
        if self.update_likelihood_sd:
            self.old_likelihood_sd.set_value(self.likelihood_sd.get_value())

    def load_prev_params(self):
        layers = filter(lambda l: isinstance(l, BayesianLayer),
                        lasagne.layers.get_all_layers(self.network)[1:])
        for layer in layers:
            layer.load_prev_params()
        if self.update_likelihood_sd:
            self.likelihood_sd.set_value(self.old_likelihood_sd.get_value())

    def get_bayesian_layers(self):
        return filter(lambda l: isinstance(l, BayesianLayer) and not l.disable_variance,
                      lasagne.layers.get_all_layers(self.network))

    def l1(self):
        layers = filter(lambda l: isinstance(l, BayesianLayer),
                        lasagne.layers.get_all_layers(self.network))
        lst_l1 = []
        num_w = 0
        for l in layers:
            l1 = l.l1_new_old()
            lst_l1.append(T.sum(l1))
            num_w += l1.shape[0]
        return sum(lst_l1) / num_w
        # return sum(l.l1_new_old() for l in layers)

    def compr_impr(self):
        """KL divergence KL[old_param||new_param]"""
        layers = filter(lambda l: isinstance(l, BayesianLayer) and not l.disable_variance,
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_old_new() for l in layers)

    def inf_gain(self):
        """KL divergence KL[new_param||old_param]"""
        layers = filter(lambda l: isinstance(l, BayesianLayer) and not l.disable_variance,
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_new_old() for l in layers)

    def num_weights(self):
        print('Disclaimer: only work with BNNLayers!')
        layers = filter(lambda l: isinstance(l, BayesianLayer), lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.num_weights() for l in layers)

    def entropy(self, input, **kwargs):
        """ Entropy approximation of a batch of input/output samples. """
        from scipy.stats import norm

        # MC samples.
        lst_pred = []
        for _ in xrange(self.n_samples):
            # Make prediction.
            lst_pred.append(self.pred_fn(input))
        arr_pred = np.asarray(lst_pred)

        # For each minibatch entry.
        lst_ent = []
        for idy in xrange(arr_pred.shape[1]):
            # Fit isotropic multivariate Gaussian function to these predictions and calculate entropy: H(s'|a,hist).
            log_var = 0.
            for idz in xrange(arr_pred.shape[2]):
                _, std = norm.fit(arr_pred[:, idy, idz])
                log_var += np.log(std ** 2)
            # I(S';\Theta) = H(s'|a,hist) - H(s'|a,hist;\theta)
            # We drop the second term as it is independent of our samples.
            ent = 0.5 * (arr_pred.shape[1] * (1 + np.log(2 * np.pi)) + log_var)
            lst_ent.append(ent)

        return np.asarray(lst_ent)

    def variance(self, input, **kwargs):
        """ Entropy approximation of a batch of input/output samples. """

        # MC samples.
        lst_pred = []
        for _ in xrange(self.n_samples):
            # Make prediction.
            pred = self.pred_fn(input)
            # if self._ind_softmax:
            # pred_rshp = pred[:, :-1].reshape((-1, self.num_classes))
            # rnd = np.random.rand(pred_rshp.shape[0])
            # pred_cumsum = np.cumsum(pred_rshp, axis=1)
            # pred_fill = np.zeros((pred_rshp.shape[0],), dtype=int)
            # for c in xrange(self.num_classes - 1):
            #     pred_fill[pred_cumsum[:, c] < rnd] = c + 1
            # pred = pred_fill.reshape((pred.shape[0], self.state_dim[1], self.state_dim[2]))
            # pred_rshp = pred[:, :-1].reshape((-1, self.num_classes))
            lst_pred.append(pred)
        arr_pred = np.asarray(lst_pred)
        return np.mean(np.var(arr_pred, axis=0), axis=1)

    def surprise(self, **kwargs):

        if self.surprise_type == ConvBNNVIME.SurpriseType.COMPR:
            surpr = self.compr_impr()
        elif self.surprise_type == ConvBNNVIME.SurpriseType.INFGAIN:
            surpr = self.inf_gain()
        elif self.surprise_type == ConvBNNVIME.SurpriseType.BALD:
            surpr = self.entropy(**kwargs)
        else:
            raise Exception('Uknown surprise_type {}'.format(self.surprise_type))
        return surpr

    def get_all_params(self):
        layers = filter(lambda l: isinstance(l, BayesianLayer), lasagne.layers.get_all_layers(self.network)[1:])
        all_mu = np.sort(np.concatenate([l.mu.eval().flatten() for l in layers]))
        all_rho = np.sort(np.concatenate([l.softplus(l.rho).eval().flatten() for l in layers]))
        return all_mu, all_rho

    def kl_div(self):
        """KL divergence KL[new_param||old_param]"""
        layers = filter(lambda l: isinstance(l, BayesianLayer) and not l.disable_variance,
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_new_old() for l in layers)

    def log_p_w_q_w_kl(self):
        """KL divergence KL[q_\phi(w)||p(w)]"""
        layers = filter(lambda l: isinstance(l, BayesianLayer) and not l.disable_variance,
                        lasagne.layers.get_all_layers(self.network))
        return sum(l.kl_div_new_prior() for l in layers)

    def reverse_log_p_w_q_w_kl(self):
        """KL divergence KL[p(w)||q_\phi(w)]"""
        layers = filter(lambda l: isinstance(l, BayesianLayer) and not l.disable_variance,
                        lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_prior_new() for l in layers)

    def _log_prob_normal(self, input, mu=0., sigma=1.):
        log_normal = - T.log(sigma) - T.log(T.sqrt(2 * np.pi)) - T.square(input - mu) / (2 * T.square(sigma))
        return T.sum(log_normal)

    def _log_prob_normal_nonsum(self, input, mu=0., sigma=1.):
        log_normal = - T.log(sigma) - T.log(T.sqrt(2 * np.pi)) - T.square(input - mu) / (2 * T.square(sigma))
        return T.sum(log_normal, axis=1)

    def pred_sym(self, input):
        return lasagne.layers.get_output(self.network, input, deterministic=False)

    def likelihood_regression(self, target, prediction, likelihood_sd):
        return self._log_prob_normal(target, prediction, likelihood_sd)

    def likelihood_regression_nonsum(self, target, prediction, likelihood_sd):
        return self._log_prob_normal_nonsum(target, prediction, likelihood_sd)

    def _log_prob_softmax(self, target, prediction):
        # Cross-entropy; target vector selecting correct prediction
        # entries.
        target = T.cast(target, 'int32')
        out_dim = np.prod(self.state_dim)
        target2 = target.reshape((prediction.shape[0], -1)) + T.arange(out_dim) * self.num_classes
        prediction_selected = prediction[:, target2]
        ll = T.sum(T.log(prediction_selected), axis=1)
        return ll

    def _log_prob_softmax_onehot(self, target, prediction):
        # Cross-entropy; target vector selecting correct prediction
        # entries.
        ll = T.sum((
            target * T.log(prediction)
        ), axis=1)
        return ll

    def likelihood_classification(self, target, prediction):
        # return T.sum(self._log_prob_softmax(target, prediction))
        target = to_onehot_sym(
            T.cast(target.flatten(), 'int32'),
            self.num_classes
        )
        target += self.label_smoothing
        target = target / target.sum(axis=1, keepdims=True)
        return T.sum(self._log_prob_softmax_onehot(
            target,
            prediction.reshape([-1, self.num_classes])
        ))

    def likelihood_classification_nonsum(self, target, prediction):
        target = to_onehot_sym(
            T.cast(target.flatten(), 'int32'),
            self.num_classes
        )
        return T.sum(self._log_prob_softmax_onehot(
            target,
            prediction.reshape([-1, self.num_classes])
        ).reshape((prediction.shape[0], -1)), axis=1)

    # def likelihood_classification_nonsum(self, target, prediction):
    #     return self._log_prob_softmax(target, prediction)

    def logp(self, input, target, **kwargs):
        log_p_D_given_w = 0.
        for _ in xrange(self.n_samples):
            if self._ind_softmax or self._logit_output:
                prediction = self.pred_sym(input)
                if not self._disable_act_rew_paths:
                    lh = self.likelihood_regression_nonsum(target[:, -1:], prediction[:, -1:], **kwargs) + \
                         self.likelihood_classification_nonsum(target[:, :-1], prediction[:, :-1])
                else:
                    lh = self.likelihood_classification_nonsum(target, prediction)
            else:
                lh = self.likelihood_regression_nonsum(target, prediction, **kwargs)
            log_p_D_given_w += lh
        return log_p_D_given_w / self.n_samples / np.prod(self.state_dim)

    def loss(self, input, target, kl_factor=1.0, disable_kl=False, **kwargs):
        if self.disable_variance:
            disable_kl = True

        # MC samples.
        log_p_D_given_w = 0.
        for _ in xrange(self.num_train_samples):
            prediction = self.pred_sym(input)
            # Calculate model likelihood log(P(D|w)).
            if self._ind_softmax or self._logit_output:
                if not self._disable_act_rew_paths:
                    lh = self.likelihood_regression(target[:, -1:], prediction[:, -1:], **kwargs) + \
                         self.likelihood_classification(target[:, :-1], prediction[:, :-1])
                else:
                    lh = self.likelihood_classification(target, prediction)
            else:
                lh = self.likelihood_regression(target, prediction, **kwargs)
            log_p_D_given_w += lh

        if disable_kl:
            return (- log_p_D_given_w / self.num_train_samples) / np.prod(self.state_dim)
        else:
            if self.update_prior:
                kl = self.kl_div()
            else:
                kl = self.log_p_w_q_w_kl()
            return (kl / self.n_batches * kl_factor - log_p_D_given_w / self.num_train_samples) / np.prod(
                self.state_dim)

    def loss_last_sample(self, input, target, **kwargs):
        """The difference with the original loss is that we only update based on the latest sample.
        This means that instead of using the prior p(w), we use the previous approximated posterior
        q(w) for the KL term in the objective function: KL[q(w)|p(w)] becomems KL[q'(w)|q(w)].
        """

        # MC samples.
        log_p_D_given_w = 0.
        for _ in xrange(self.n_samples):
            prediction = self.pred_sym(input)
            # Calculate model likelihood log(P(D|w)).
            if self._ind_softmax:
                if not self._disable_act_rew_paths:
                    lh = self.likelihood_regression(target[:, -1:], prediction[:, -1:], **kwargs) + \
                         self.likelihood_classification(target[:, :-1], prediction[:, :-1])
                else:
                    lh = self.likelihood_classification(target, prediction)
            else:
                lh = self.likelihood_regression(target, prediction, **kwargs)
            log_p_D_given_w += lh

        return (self.kl_div() - log_p_D_given_w / self.n_samples) / np.prod(self.state_dim)

    def build_network(self):

        # Make sure that we are able to unmerge the s_in and a_in.

        # Input to the s_net is always flattened.
        input_dim = (self.num_seq_inputs,) + (self.state_dim[1:])
        s_flat_dim = np.prod(input_dim)

        if self._logit_output:
            self._logit_s = theano.shared(0.1, name='logit_s')

        if not self._disable_act_rew_paths:
            print('f: {} x {} -> {} x {}'.format(input_dim, self.action_dim, self.state_dim, self.reward_dim))
            a_flat_dim = np.prod(self.action_dim)
            r_flat_dim = np.prod(self.reward_dim)
            input = lasagne.layers.InputLayer(shape=(None, s_flat_dim + a_flat_dim))
            # Split input into state and action.
            s_net = lasagne.layers.SliceLayer(input, indices=slice(None, s_flat_dim), axis=1)
            a_net = lasagne.layers.SliceLayer(input, indices=slice(s_flat_dim, None), axis=1)
            r_net = None
            a_net = lasagne.layers.reshape(a_net, ([0],) + self.action_dim)
            print('Slicing into {} and {}'.format(s_net.output_shape, a_net.output_shape))
        else:
            print('f: {} -> {}'.format(input_dim, self.state_dim))
            s_net = lasagne.layers.InputLayer(shape=(None, s_flat_dim))

        # Reshape according to the input_dim
        s_net = lasagne.layers.reshape(s_net, ([0],) + input_dim)
        # FIXME: magic number
        dropout_p = 0.5
        for i, layer_disc in enumerate(self.layers_disc):

            if 'nonlinearity' in layer_disc.keys() and layer_disc['nonlinearity'] == 'sin':
                def sinsq(x):
                    return np.sin(np.square(x))

                layer_disc['nonlinearity'] = sinsq

            if layer_disc['name'] == 'convolution':
                s_net = BayesianConvLayer(
                    s_net,
                    num_filters=layer_disc['n_filters'],
                    filter_size=layer_disc['filter_size'],
                    nonlinearity=layer_disc['nonlinearity'],
                    prior_sd=self.prior_sd,
                    pad=layer_disc['pad'],
                    stride=layer_disc['stride'],
                    disable_variance=layer_disc['deterministic'],
                    logit_weights=self._logit_weights)
                if layer_disc['dropout'] is True:
                    s_net = dropout(s_net, p=dropout_p)
                if layer_disc['batch_norm'] is True:
                    s_net = batch_norm(s_net)
            elif layer_disc['name'] == 'gaussian':
                if 'nonlinearity' not in layer_disc.keys():
                    layer_disc['nonlinearity'] = lasagne.nonlinearities.rectify
                s_net = BayesianDenseLayer(
                    s_net, num_units=layer_disc['n_units'],
                    nonlinearity=layer_disc['nonlinearity'],
                    prior_sd=self.prior_sd,
                    use_local_reparametrization_trick=self.use_local_reparametrization_trick,
                    disable_variance=layer_disc['deterministic'],
                    matrix_variate_gaussian=layer_disc['matrix_variate_gaussian'],
                    logit_weights=self._logit_weights)
                if layer_disc['dropout'] is True:
                    s_net = dropout(s_net, p=dropout_p)
                if layer_disc['batch_norm'] is True:
                    s_net = batch_norm(s_net)
            elif layer_disc['name'] == 'discrete_embedding':
                if 'nonlinearity' not in layer_disc.keys():
                    layer_disc['nonlinearity'] = lasagne.nonlinearities.rectify
                s_net = DiscreteEmbeddingLayer(
                    s_net, num_units=layer_disc['n_units'],
                    nonlinearity=layer_disc['nonlinearity'],
                    prior_sd=self.prior_sd,
                    use_local_reparametrization_trick=self.use_local_reparametrization_trick,
                    disable_variance=layer_disc['deterministic'],
                    matrix_variate_gaussian=layer_disc['matrix_variate_gaussian'],
                    logit_weights=self._logit_weights)
                if layer_disc['dropout'] is True:
                    s_net = dropout(s_net, p=dropout_p)
                if layer_disc['batch_norm'] is True:
                    s_net = batch_norm(s_net)
            elif layer_disc['name'] == 'deterministic':
                s_net = lasagne.layers.DenseLayer(
                    s_net,
                    num_units=layer_disc['n_units'],
                    nonlinearity=self.transf)
                if layer_disc['batch_norm'] is True:
                    s_net = batch_norm(s_net, p=dropout_p)
            elif layer_disc['name'] == 'deconvolution':
                s_net = BayesianDeConvLayer(
                    s_net, num_filters=layer_disc['n_filters'],
                    filter_size=layer_disc['filter_size'],
                    prior_sd=self.prior_sd,
                    stride=layer_disc['stride'],
                    crop=layer_disc['pad'],
                    disable_variance=layer_disc['deterministic'],
                    nonlinearity=layer_disc['nonlinearity'],
                    logit_weights=self._logit_weights)
                if layer_disc['dropout'] is True:
                    s_net = dropout(s_net, p=dropout_p)
                if layer_disc['batch_norm'] is True:
                    s_net = batch_norm(s_net)
            elif layer_disc['name'] == 'reshape':
                s_net = lasagne.layers.ReshapeLayer(
                    s_net,
                    shape=layer_disc['shape'])
            elif layer_disc['name'] == 'pool':
                s_net = lasagne.layers.Pool2DLayer(
                    s_net,
                    pool_size=layer_disc['pool_size'])
            elif layer_disc['name'] == 'upscale':
                s_net = lasagne.layers.Upscale2DLayer(
                    s_net,
                    scale_factor=layer_disc['scale_factor'])
            elif layer_disc['name'] == 'hadamard':
                if 'nonlinearity' not in layer_disc.keys():
                    layer_disc['nonlinearity'] = lasagne.nonlinearities.rectify
                a_net = BayesianDenseLayer(
                    s_net, num_units=layer_disc['n_units'],
                    nonlinearity=layer_disc['nonlinearity'],
                    prior_sd=self.prior_sd,
                    use_local_reparametrization_trick=self.use_local_reparametrization_trick,
                    disable_variance=layer_disc['deterministic'],
                    matrix_variate_gaussian=layer_disc['matrix_variate_gaussian'])
                if layer_disc['dropout'] is True:
                    a_net = dropout(a_net, p=dropout_p)
                s_net = HadamardLayer([s_net, a_net])
                if layer_disc['batch_norm'] is True:
                    s_net = batch_norm(s_net)
            elif layer_disc['name'] == 'outerprod':
                # Here we fuse the s_net with the a_net through an outer
                # product.
                s_net = lasagne.layers.flatten(OuterProdLayer([s_net, a_net]))
            elif layer_disc['name'] == 'split':
                # Split off the r_net from s_net.
                r_net = BayesianDenseLayer(
                    s_net,
                    num_units=layer_disc['n_units'],
                    nonlinearity=self.transf,
                    prior_sd=self.prior_sd,
                    use_local_reparametrization_trick=self.use_local_reparametrization_trick,
                    disable_variance=layer_disc['deterministic'],
                    matrix_variate_gaussian=layer_disc['matrix_variate_gaussian'])
                if layer_disc['batch_norm'] is True:
                    r_net = batch_norm(r_net)
            else:
                raise (Exception('Unknown layer!'))

            print('layer {}: {}\n\toutsize: {}'.format(
                i, layer_disc, s_net.output_shape))
            if not self._disable_act_rew_paths and a_net is not None:
                print('\toutsize: {}'.format(a_net.output_shape))
            if not self._disable_act_rew_paths and r_net is not None:
                print('\toutsize: {}'.format(r_net.output_shape))

        # Output of output_dim is flattened again. But ofc, we need to output
        # the r_net value, e.g., the reward signal.
        if self._ind_softmax:
            s_net = IndependentSoftmaxLayer(
                s_net,
                num_bins=self.num_classes,
            )
            print('layer ind softmax:\n\toutsize: {}'.format(s_net.output_shape))
            s_net = lasagne.layers.reshape(s_net, ([0], -1))
        elif self._logit_output:
            s_net = LogitLayer(
                s_net,
                num_bins=self.num_classes,
                s=self._logit_s
            )
            s_net = lasagne.layers.reshape(s_net, ([0], -1))
        else:
            s_net = lasagne.layers.reshape(s_net, ([0], -1))

        if not self._disable_act_rew_paths:
            r_net = BayesianDenseLayer(
                r_net,
                num_units=r_flat_dim,
                nonlinearity=lasagne.nonlinearities.linear,
                prior_sd=self.prior_sd,
                disable_variance=True,
                use_local_reparametrization_trick=self.use_local_reparametrization_trick,
                matrix_variate_gaussian=False)
            r_net = lasagne.layers.reshape(r_net, ([0], -1))
            self.network = ConcatLayer([s_net, r_net])
        else:
            self.network = s_net

    def build_model(self):

        # Prepare Theano variables for inputs and targets
        # Same input for classification as regression.
        kl_factor = T.scalar('kl_factor',
                             dtype=theano.config.floatX)
        # Assume all inputs are flattened.
        input_var = T.matrix('inputs',
                             dtype=theano.config.floatX)

        target_var = T.matrix('targets',
                              dtype=theano.config.floatX)

        # Make the likelihood standard deviation a trainable parameter.
        self.likelihood_sd = theano.shared(
            value=self.likelihood_sd_init,  # self.likelihood_sd_init,
            name='likelihood_sd'
        )
        self.old_likelihood_sd = theano.shared(
            value=self.likelihood_sd_init,  # self.likelihood_sd_init,
            name='old_likelihood_sd'
        )

        # Loss function.
        loss = self.loss(
            input_var, target_var, kl_factor, likelihood_sd=self.likelihood_sd)
        loss_only_last_sample = self.loss_last_sample(
            input_var, target_var, likelihood_sd=self.likelihood_sd)

        # Create update methods.
        params_kl = lasagne.layers.get_all_params(self.network, trainable=True)
        params = []
        params.extend(params_kl)
        if self.output_type == 'regression' and self.update_likelihood_sd:
            # No likelihood sd for classification tasks.
            params.append(self.likelihood_sd)
        if self._logit_output:
            params.append(self._logit_s)

        # Train/val fn.
        self.pred_fn = ext.compile_function(
            [input_var], self.pred_sym(input_var), log_name='fn_pred')

        updates = lasagne.updates.adam(
            loss, params, learning_rate=self.learning_rate)

        # We want to resample when actually updating the BNN itself, otherwise
        # you will fit to the specific noise.
        self.train_fn = ext.compile_function(
            [input_var, target_var, kl_factor], loss, updates=updates, log_name='fn_train')

        # self.fn_loss = ext.compile_function(
        #     [input_var, target_var, kl_factor], loss, log_name='fn_loss')

        # Surprise functions:
        # INFGAIN: useable with 2nd_order_update = True/False
        # BALD: useable
        # COMPGAIN: useable with 2nd_order_update = True/False,
        # needs to be calculated using logp, not KL!
        # ---------------------------------------

        def second_order_update(loss, params, oldparams, step_size):
            """Second-order update method for optimizing loss_last_sample, so basically,
            KL term (new params || old params) + NLL of latest sample. The Hessian is
            evaluated at the origin and provides curvature information to make a more
            informed step in the correct descent direction."""
            grads = theano.grad(loss, params)
            updates = OrderedDict()

            for i in xrange(len(params)):
                param = params[i]
                grad = grads[i]

                if param.name == 'mu' or param.name == 'b_mu':
                    oldparam_rho = oldparams[i + 1]
                    invH = T.square(T.log(1 + T.exp(oldparam_rho)))
                elif param.name == 'rho' or param.name == 'b_rho':
                    oldparam_rho = oldparams[i]
                    p = param
                    H = 2. * (T.exp(2 * p)) / (1 + T.exp(p)) ** 2 / (T.log(1 + T.exp(p)) ** 2)
                    invH = 1. / H
                elif param.name == 'likelihood_sd':
                    invH = 0.
                updates[param] = param - T.cast(step_size * invH * grad, 'float32')

            return updates

        if self.surprise_type == ConvBNNVIME.SurpriseType.INFGAIN:
            if self.second_order_update:

                step_size = T.scalar('step_size',
                                     dtype=theano.config.floatX)

                def fast_kl_div(loss, params, step_size):
                    # FIXME: doesn't work with MVG.

                    grads = T.grad(loss, params)

                    kl_component = []
                    for i in xrange(len(params)):

                        param = params[i]
                        grad = grads[i]

                        if param.name == 'mu' or param.name == 'b_mu':
                            oldparam_rho = params[i + 1]
                            invH = T.square(T.log(1 + T.exp(oldparam_rho)))
                        elif param.name == 'rho' or param.name == 'b_rho':
                            p = param
                            H = 2. * (T.exp(2 * p)) / (1 + T.exp(p)) ** 2 / (T.log(1 + T.exp(p)) ** 2)
                            invH = 1. / H
                        elif param.name == 'likelihood_sd':
                            invH = 0.

                        kl_component.append(T.sum(0.5 * T.square(step_size) * T.square(grad) * invH))

                    return sum(kl_component)

                params_bayesian = []
                # sel_layers = filter(lambda l: isinstance(l, BayesianLayer) and not l.disable_variance,
                #                     lasagne.layers.get_all_layers(self.network))
                # params_bayesian.extend(sel_layers[-1].get_params(trainable=True, bayesian=True))
                params_bayesian.extend(lasagne.layers.get_all_params(self.network, trainable=True, bayesian=True))
                if self.output_type == 'regression' and self.update_likelihood_sd:
                    params_bayesian.append(self.likelihood_sd)
                compute_fast_kl_div = fast_kl_div(
                    loss_only_last_sample, params_bayesian, step_size)
                self.train_update_fn = ext.compile_function(
                    [input_var, target_var, step_size], compute_fast_kl_div, log_name='fn_surprise_fast',
                    no_default_updates=False)

                # Code to actually perform second order updates
                # ---------------------------------------------
                #             updates_kl = second_order_update(
                #                 loss_only_last_sample, params, oldparams, step_size)
                #
                #             self.train_update_fn = ext.compile_function(
                #                 [input_var, target_var, step_size], loss_only_last_sample, updates=updates_kl,
                # log_name='fn_surprise_2nd', no_default_updates=False)
                # ---------------------------------------------

            else:
                # Use SGD to update the model for a single sample, in order to
                # calculate the surprise. We only update the bayesian params, no batchnorm params.
                # params_bayesian = []
                # params_bayesian.extend(lasagne.layers.get_all_params(self.network, trainable=True, bayesian=True))
                # if self.output_type == 'regression' and self.update_likelihood_sd:
                #     params_bayesian.append(self.likelihood_sd)
                # Use the loss with KL[post'||post'].
                loss_infgain = self.loss_last_sample(
                    input_var, target_var, likelihood_sd=self.likelihood_sd)
                # SGD rather than adam, we don't want momentum. Learning rate tuned for adam, needs to be smaller.
                updates_infgain = lasagne.updates.adam(
                    loss_infgain, params, learning_rate=self.learning_rate)
                self.train_update_fn = ext.compile_function(
                    [input_var, target_var, kl_factor], loss_infgain, updates=updates_infgain,
                    log_name='fn_train_infgain_1storder',
                    no_default_updates=False)
                # Need explicit kl calc.
                self.fn_kl = ext.compile_function(
                    [], self.kl_div(), log_name='fn_kl'
                )

        elif self.surprise_type == ConvBNNVIME.SurpriseType.BALD:
            # BALD: I(S';\Theta|a,hist) = H(S'|a,hist) - H(S'|a,hist;\theta)
            self.train_update_fn = self.entropy

        elif self.surprise_type == ConvBNNVIME.SurpriseType.VAR:
            # Focus on maximum variance: basically, when we rewrite BALD as only the first entropy term,
            # we can see it as focusing on variance.
            self.train_update_fn = self.variance

        elif self.surprise_type == ConvBNNVIME.SurpriseType.L1:
            # Use SGD to update the model for a single sample, in order to
            # calculate the surprise. We only update the bayesian params, no batchnorm params.
            params_bayesian = []
            # FIXME
            print('BAYESIAN=TRUE flag wont work for deterministic NN')
            print('BAYESIAN=TRUE flag wont work for deterministic NN')
            print('BAYESIAN=TRUE flag wont work for deterministic NN')
            params_bayesian.extend(lasagne.layers.get_all_params(self.network, trainable=True, bayesian=True))
            # Use the loss with KL[post'||post'].
            loss_l1 = self.loss(
                input_var, target_var, disable_kl=True, likelihood_sd=self.likelihood_sd)
            # SGD rather than adam, we don't want momentum. Learning rate tuned for adam, needs to be smaller.
            updates_l1 = lasagne.updates.adam(
                loss_l1, params_bayesian, learning_rate=self.learning_rate)
            self.train_update_fn = ext.compile_function(
                [input_var, target_var, kl_factor], loss_l1, updates=updates_l1,
                log_name='fn_train_l1',
                no_default_updates=False)
            self.fn_l1 = ext.compile_function(
                [], self.l1(), log_name='fn_l1'
            )

        elif self.surprise_type == ConvBNNVIME.SurpriseType.COMPR:
            # COMPR IMPR (no KL)
            # Calculate logp.
            self.fn_logp = ext.compile_function(
                [input_var, target_var], self.logp(input_var, target_var, likelihood_sd=self.likelihood_sd),
                log_name='fn_logp')
            if self.second_order_update:
                oldparams = lasagne.layers.get_all_params(
                    self.network, oldparam=True)
                step_size = T.scalar('step_size',
                                     dtype=theano.config.floatX)
                params_bayesian = []
                params_bayesian.extend(lasagne.layers.get_all_params(self.network, trainable=True, bayesian=True))

                updates_bayesian = second_order_update(
                    loss_only_last_sample, params_bayesian, oldparams, step_size)

                self.train_update_fn = ext.compile_function(
                    [input_var, target_var, step_size], loss_only_last_sample, updates=updates_bayesian,
                    log_name='fn_surprise_2nd', no_default_updates=False)


if __name__ == '__main__':
    pass
