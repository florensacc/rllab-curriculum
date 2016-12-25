import numpy as np
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as NL
import theano
import theano.tensor as TT
from rllab.misc.ext import compile_function
from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.network import ConvNetwork
from rllab.misc import tensor_utils
from rllab.optimizers.lbfgs_optimizer import LbfgsOptimizer
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.core.serializable import Serializable
from rllab.misc.ext import iterate_minibatches_generic
from rllab.misc import logger


class ConvQFunction(LasagnePowered, Serializable):
    """
    Computes a value for each observation (vector) and action (integer).
    Assumes discrete actions.

    TODO: normalize outputs
    """

    def __init__(
            self,
            name,
            input_shape,
            output_dim,
            hidden_sizes,
            conv_filters,conv_filter_sizes,conv_strides,conv_pads,
            hidden_nonlinearity=NL.rectify,

            optimizer=None,
            subsample_factor=1.0,
            batchsize=None,
            max_train_batch=100,
            debug=False,
    ):
        """
        :param input_shape: usually for images of the form (width,height,channel)
        :param output_dim: Dimension of output.
        :param hidden_sizes: Number of hidden units of each layer of the mean network.
        :param hidden_nonlinearity: Non-linearity used for each layer of the mean network.
        :param optimizer: Optimizer for minimizing the negative log-likelihood.
        :param use_trust_region: Whether to use trust region constraint.
        :param step_size: KL divergence constraint for each iteration
        :param learn_std: Whether to learn the standard deviations. Only effective if adaptive_std is False. If
        adaptive_std is True, this parameter is ignored, and the weights for the std network are always learned.
        :param adaptive_std: Whether to make the std a function of the states.
        :param std_share_network: Whether to use the same network as the mean.
        :param std_hidden_sizes: Number of hidden units of each layer of the std network. Only used if
        `std_share_network` is False. It defaults to the same architecture as the mean.
        :param std_nonlinearity: Non-linearity used for each layer of the std network. Only used if `std_share_network`
        is False. It defaults to the same non-linearity as the mean.
        """
        Serializable.quick_init(self, locals())

        if optimizer is None:
            if use_trust_region:
                optimizer = PenaltyLbfgsOptimizer("optimizer")
            else:
                optimizer = LbfgsOptimizer("optimizer")

        self._optimizer = optimizer

        self.input_shape = input_shape
        self.output_dim = output_dim
        network = ConvNetwork(
            name="network",
            input_shape=input_shape,
            output_dim=output_dim,
            conv_filters=conv_filters,
            conv_filter_sizes=conv_filter_sizes,
            conv_strides=conv_strides,
            conv_pads=conv_pads,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=None,
        )

        l_qs = network.output_layer # N x |A|
        all_qs_var = L.get_output(l_qs)


        LasagnePowered.__init__(self, [l_qs])

        observations_var = network.input_layer.input_var
        actions_var = TT.ivector("actions")
        qs_var = all_qs_var[:,actions_var] # N
        ys_var = TT.fvector("ys")
        loss = TT.mean((qs_var - ys_var)**2)

        self._f_all_qs = compile_function([observations_var], all_qs_var)
        self._f_qs = compile_function([observations_var, actions_var],qs_var)
        self._f_loss = compile_function([observations_var, actions_var, ys_var], loss)
        self._l_qs = all_qs_var

        # tell the optimizer
        optimizer_args = dict(
            loss=loss,
            target=self,
            inputs=[observations_var, actions_var, ys_var],
        )
        self._optimizer.update_opt(**optimizer_args)

        self._name = name

        self._network = network
        self._subsample_factor = subsample_factor
        self._batchsize = batchsize
        self._max_train_batch = max_train_batch
        self.debug = debug

    def fit(self, obs, actions, ys):

        if self._subsample_factor < 1:
            num_samples_tot = obs.shape[0]
            idx = np.random.randint(0, num_samples_tot, int(num_samples_tot * self._subsample_factor))
            obs, actions, ys = obs[idx], actions[idx], ys[idx]

        if self._name:
            prefix = self._name + "_"
        else:
            prefix = ""

        inputs = (obs, actions, ys)
        loss_before = self._optimizer.loss(inputs)

        # randomly sample a minibatch each time
        n_inputs = len(ys)
        for iteration in range(self._max_train_batch):
            indices = np.random.randint(
                low=0, high=n_inputs,
                size=self._batchsize)
            _obs = obs[indices]
            _actions = actions[indices]
            _ys = ys[indices]
            _inputs = [_obs, _actions, _ys]
            self._optimizer.optimize(_inputs)

            if self.debug:
                cur_loss = self._optimizer.loss(inputs)
                if self.rank == 0:
                    logger.log('qf iteration: %d, loss: %f'%(
                        iteration,
                        cur_loss,
                    ))

        loss_after = self._optimizer.loss(inputs)
        logger.record_tabular(prefix + 'LossBefore', loss_before)
        logger.record_tabular(prefix + 'LossAfter', loss_after)
        logger.record_tabular(prefix + 'dLoss', loss_before - loss_after)

    def compute_qs(self, observations, actions):
        return self._f_qs(observations, actions)

    def compute_all_qs(self, observations):
        return self._f_all_qs(observations)

    def get_param_values(self, **tags):
        return LasagnePowered.get_param_values(self, **tags)

    def set_param_values(self, flattened_params, **tags):
        return LasagnePowered.set_param_values(self, flattened_params, **tags)

    def init_par_objs(self, n_parallel):
        size_grad = len(self.get_param_values(trainable=True))
        self._optimizer.init_par_objs(n_parallel, size_grad)

    def init_rank(self, rank):
        self.rank = rank
        self._optimizer.init_rank(rank)

    def force_compile(self):
        observations = np.zeros((1,np.prod(self.input_shape))).astype(theano.config.floatX)
        actions = np.zeros(1)
        ys = np.zeros(1).astype(theano.config.floatX)
        inputs = (observations, actions, ys)
        self._optimizer.force_compile(inputs)
