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
from sandbox.rein.dynamics_models.utils import enum
from sandbox.rein.dynamics_models.bnn.conv_bnn import BayesianConvLayer, BayesianDeConvLayer, BayesianDenseLayer, \
    BayesianLayer


class IndependentSoftmaxLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_bins, W=lasagne.init.GlorotUniform(), **kwargs):
        super(IndependentSoftmaxLayer, self).__init__(incoming, **kwargs)

        self._num_bins = num_bins
        self.W = self.add_param(W, (self.input_shape[1], self._num_bins), name="W")

    def get_output_for(self, input, **kwargs):
        _a = T.dot(input.dimshuffle(0, 2, 3, 1), self.W)
        _b = T.exp(_a - T.max(_a, axis=3).dimshuffle(0, 1, 2, 'x'))
        _c = T.sum(_b, axis=3).dimshuffle(0, 1, 2, 'x')
        return T.clip(_b / _c, 1e-8, 1)

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
        INFGAIN='information gain', COMPR='compression gain', BALD='BALD')

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
                 disable_act_rew_paths=False  # Disable action and reward modeling, just s -> s' prediction.
                 ):

        Serializable.quick_init(self, locals())

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim

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
        layers = filter(lambda l: isinstance(l, BayesianLayer), lasagne.layers.get_all_layers(self.network)[1:])
        for layer in layers:
            layer.load_prev_params()
        if self.update_likelihood_sd:
            self.likelihood_sd.set_value(self.old_likelihood_sd.get_value())

    def compr_impr(self):
        """KL divergence KL[old_param||new_param]"""
        layers = filter(lambda l: isinstance(l, BayesianLayer), lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_old_new() for l in layers)

    def inf_gain(self):
        """KL divergence KL[new_param||old_param]"""
        layers = filter(lambda l: isinstance(l, BayesianLayer), lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_new_old() for l in layers)

    def num_weights(self):
        print('Disclaimer: only work with BNNLayers!')
        layers = filter(lambda l: isinstance(l, BayesianLayer), lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.num_weights() for l in layers)

    def ent(self, input):
        # FIXME: work in progress
        mtrx_pred = np.zeros((self.n_samples, self.n_out))
        for i in xrange(self.n_samples):
            # Make prediction.
            mtrx_pred[i] = self.pred_fn(input)
        cov = np.cov(mtrx_pred, rowvar=0)
        if isinstance(cov, float):
            var = np.trace(cov) / float(cov.shape[0])
        else:
            var = cov
        return var

    def entropy(self, input, likelihood_sd, **kwargs):
        """ Entropy of a batch of input/output samples. """

        # MC samples.
        _log_p_D_given_w = []
        for _ in xrange(self.n_samples):
            # Make prediction.
            prediction = self.pred_sym(input)
            for _ in xrange(self.n_samples):
                sampled_mean = self.pred_sym(input)
                # Calculate model likelihood log(P(D|w)).
                if self.output_type == ConvBNNVIME.OutputType.CLASSIFICATION:
                    lh = self.likelihood_classification(sampled_mean, prediction)
                elif self.output_type == ConvBNNVIME.OutputType.REGRESSION:
                    lh = self.likelihood_regression(sampled_mean, prediction, likelihood_sd)
                _log_p_D_given_w.append(lh)
        log_p_D_given_w = sum(_log_p_D_given_w)

        return - log_p_D_given_w / (self.n_samples) ** 2 + 0.5 * (np.log(2 * np.pi * likelihood_sd ** 2) + 1)

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
        layers = filter(lambda l: isinstance(l, BayesianLayer), lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_new_old() for l in layers)

    def log_p_w_q_w_kl(self):
        """KL divergence KL[q_\phi(w)||p(w)]"""
        layers = filter(lambda l: isinstance(l, BayesianLayer), lasagne.layers.get_all_layers(self.network)[1:])
        return sum(l.kl_div_new_prior() for l in layers)

    def reverse_log_p_w_q_w_kl(self):
        """KL divergence KL[p(w)||q_\phi(w)]"""
        layers = filter(lambda l: isinstance(l, BayesianLayer), lasagne.layers.get_all_layers(self.network)[1:])
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

    def likelihood_classification(self, target, prediction):
        return T.sum(self._log_prob_softmax(target, prediction))

    def likelihood_classification_nonsum(self, target, prediction):
        return self._log_prob_softmax(target, prediction)

    def logp(self, input, target, **kwargs):
        if not self._ind_softmax:
            log_p_D_given_w = 0.
            for _ in xrange(self.n_samples):
                prediction = self.pred_sym(input)
                lh = self.likelihood_regression_nonsum(target, prediction, **kwargs)
                log_p_D_given_w += lh
            return log_p_D_given_w / self.n_samples
        else:
            log_p_D_given_w = 0.
            for _ in xrange(self.n_samples):
                pred = self.pred_sym(input)
                if not self._disable_act_rew_paths:
                    lh = self.likelihood_regression(target[:, -1], pred[:, -1], **kwargs) + \
                         self.likelihood_classification_nonsum(target[:, :-1], pred[:, :-1])
                else:
                    lh = self.likelihood_classification_nonsum(target, pred)
                log_p_D_given_w += lh
            return log_p_D_given_w / self.n_samples

    def loss(self, input, target, kl_factor=1.0, disable_kl=False, **kwargs):
        if self.disable_variance:
            disable_kl = True
        # MC samples.
        log_p_D_given_w = 0.
        for _ in xrange(self.num_train_samples):
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

        if disable_kl:
            return - log_p_D_given_w / self.num_train_samples
        else:
            if self.update_prior:
                kl = self.kl_div()
            else:
                kl = self.log_p_w_q_w_kl()
            return kl / self.n_batches * kl_factor - log_p_D_given_w / self.num_train_samples

    def loss_last_sample(self, input, target, **kwargs):
        """The difference with the original loss is that we only update based on the latest sample.
        This means that instead of using the prior p(w), we use the previous approximated posterior
        q(w) for the KL term in the objective function: KL[q(w)|p(w)] becomems KL[q'(w)|q(w)].
        """
        # TODO: check out if this is still correct.
        return self.loss(input, target, disable_kl=True, **kwargs)

    def build_network(self):

        # Make sure that we are able to unmerge the s_in and a_in.

        # Input to the s_net is always flattened.
        s_flat_dim = np.prod(self.state_dim)

        if not self._disable_act_rew_paths:
            print('f: {} x {} -> {} x {}'.format(self.state_dim, self.action_dim, self.state_dim, self.reward_dim))
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
            print('f: {} -> {}'.format(self.state_dim, self.state_dim))
            s_net = lasagne.layers.InputLayer(shape=(None, s_flat_dim))

        # Reshape according to the input_dim
        s_net = lasagne.layers.reshape(s_net, ([0],) + self.state_dim)
        # FIXME: magic number
        dropout_p = 0.5
        for i, layer_disc in enumerate(self.layers_disc):

            if layer_disc['name'] == 'convolution':
                s_net = BayesianConvLayer(
                    s_net,
                    num_filters=layer_disc['n_filters'],
                    filter_size=layer_disc['filter_size'],
                    prior_sd=self.prior_sd,
                    pad=layer_disc['pad'],
                    stride=layer_disc['stride'],
                    disable_variance=self.disable_variance)
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
                    disable_variance=self.disable_variance,
                    matrix_variate_gaussian=layer_disc['matrix_variate_gaussian'])
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
                    disable_variance=self.disable_variance,
                    nonlinearity=layer_disc['nonlinearity'])
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
                    disable_variance=self.disable_variance,
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
                    disable_variance=self.disable_variance,
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
        else:
            s_net = lasagne.layers.reshape(s_net, ([0], -1))

        if not self._disable_act_rew_paths:
            r_net = BayesianDenseLayer(
                r_net,
                num_units=r_flat_dim,
                nonlinearity=lasagne.nonlinearities.linear,
                prior_sd=self.prior_sd,
                disable_variance=self.disable_variance,
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

        if self.output_type == ConvBNNVIME.OutputType.REGRESSION:
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

        elif self.output_type == ConvBNNVIME.OutputType.CLASSIFICATION:

            target_var = T.matrix('targets', dtype=theano.config.floatX)

            # Loss function.
            loss = self.loss(
                input_var, target_var, kl_factor, disable_kl=self.disable_variance)
            loss_only_last_sample = self.loss_last_sample(
                input_var, target_var)

        else:
            raise Exception(
                'Unknown self.output_type {}'.format(self.output_type))

        # Create update methods.
        params_kl = lasagne.layers.get_all_params(self.network, trainable=True)
        params = []
        params.extend(params_kl)
        if self.output_type == 'regression' and self.update_likelihood_sd:
            # No likelihood sd for classification tasks.
            params.append(self.likelihood_sd)

        # Train/val fn.
        self.pred_fn = ext.compile_function(
            [input_var], self.pred_sym(input_var), log_name='fn_pred')

        if self.update_prior:
            # When using posterior chaining, prefer SGD as we dont want to build up
            # momentum between prior-posterior updates.
            def sgd_clip(loss, params, learning_rate):
                grads = theano.grad(loss, params)
                updates = OrderedDict()
                for param, grad in zip(params, grads):
                    updates[param] = param - learning_rate * T.clip(grad, -1., 1.)
                return updates

            # Clipping grads seems necessary to prevent explosion.
            updates = sgd_clip(
                loss, params, learning_rate=self.learning_rate)
        else:
            updates = lasagne.updates.adam(
                loss, params, learning_rate=self.learning_rate)

        # We want to resample when actually updating the BNN itself, otherwise
        # you will fit to the specific noise.
        self.train_fn = ext.compile_function(
            [input_var, target_var, kl_factor], loss, updates=updates, log_name='fn_train')
        self.fn_loss = ext.compile_function(
            [input_var, target_var, kl_factor], loss, log_name='fn_loss')

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
                updates[param] = param - step_size * invH * grad

            return updates

        if self.surprise_type == ConvBNNVIME.SurpriseType.INFGAIN:
            if self.second_order_update:

                oldparams = lasagne.layers.get_all_params(
                    self.network, oldparam=True)
                step_size = T.scalar('step_size',
                                     dtype=theano.config.floatX)

                def fast_kl_div(loss, params, oldparams, step_size):
                    # FIXME: doesn't work with MVG.

                    grads = T.grad(loss, params)

                    kl_component = []
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

                        kl_component.append(T.sum(0.5 * T.square(step_size) * T.square(grad) * invH))

                    return sum(kl_component)

                compute_fast_kl_div = fast_kl_div(
                    loss_only_last_sample, params, oldparams, step_size)

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
                # calculate the surprise.
                self.fn_kl = ext.compile_function(
                    [], self.kl_div(), log_name='fn_kl'
                )

        elif self.surprise_type == ConvBNNVIME.SurpriseType.BALD:
            # BALD
            self.train_update_fn = ext.compile_function(
                [input_var], self.surprise(input=input_var, likelihood_sd=self.likelihood_sd),
                log_name='fn_surprise_bald')
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
                updates_kl = second_order_update(
                    loss_only_last_sample, params, oldparams, step_size)

                self.train_update_fn = ext.compile_function(
                    [input_var, target_var, step_size], loss_only_last_sample, updates=updates_kl,
                    log_name='fn_surprise_2nd', no_default_updates=False)


if __name__ == '__main__':
    pass
