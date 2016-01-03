import lasagne
import lasagne.layers as L
import rllab.core.lasagne_recurrent as LR
import rllab.core.lasagne_helpers as LH
import lasagne.nonlinearities as NL
import numpy as np
import theano
import theano.tensor as TT
from rllab.core.lasagne_layers import ParamLayer
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.misc.ext import compile_function, merge_dict, new_tensor
from rllab.policy.base import StochasticPolicy
from rllab.misc.overrides import overrides


def log_normal_pdf(x, mean, log_std):
    normalized = (x - mean) / TT.exp(log_std)
    return -0.5*TT.square(normalized) - np.log((2*np.pi)**0.5) - log_std


class MeanStdRNNPolicy(StochasticPolicy, LasagnePowered, Serializable):

    def __init__(self, mdp):
        # create network

        n_hidden = 10  # 32
        grad_clip = 100

        # forget_gate = LR.Gate(b=lasagne.init.Constant(5.0))

        l_in = L.InputLayer(shape=(None, None, mdp.observation_shape[0]))

        # These four definitions below are more or less dummy variables and
        # layers. We'd like to support plugging in the initial values of hidden
        # and cell states to support computation on-the-fly when rolling out
        # policies; on the other hand, when doing training we don't want to
        # have to pass in zero state values as inputs. Hence we treat them as
        # shared variables here with zero value. When we need to pass in the
        # values, we replace the input layers with other tensor variables
        hid_init_1 = theano.shared(
            np.zeros((1, n_hidden), dtype=theano.config.floatX),
            name="hid_init_1",
        )
        cell_init_1 = theano.shared(
            np.zeros((1, n_hidden), dtype=theano.config.floatX),
            name="cell_init_1",
        )
        l_hid_init_1 = L.InputLayer(
            input_var=hid_init_1, shape=(1, n_hidden))
        l_cell_init_1 = L.InputLayer(
            input_var=cell_init_1, shape=(1, n_hidden))

        reset_var = theano.shared(
            np.zeros((1, 1), dtype='uint8'),
            name="reset",
        )
        l_reset = L.InputLayer(
            input_var=reset_var, shape=(None, None)
        )

        l_forward_1 = LR.LSTMLayer(
            l_in,
            hid_init=l_hid_init_1,
            cell_init=l_cell_init_1,
            num_units=n_hidden,
            reset_input=l_reset,
            grad_clipping=grad_clip,
            nonlinearity=NL.tanh,
            # forgetgate=forget_gate,
        )

        l_forward_reshaped = L.ReshapeLayer(l_forward_1, (-1, n_hidden))

        l_mean = L.DenseLayer(
            l_forward_reshaped,
            num_units=mdp.action_dim,
            nonlinearity=NL.tanh
        )
        l_log_std = ParamLayer(
            l_forward_reshaped,
            num_units=mdp.action_dim
        )

        hid_1_var = TT.matrix('hid_1')
        cell_1_var = TT.matrix('cell_1')

        outputs, extra_outputs = \
            LH.get_output_with_extra(
                [l_mean, l_log_std, l_forward_1],
                {l_hid_init_1: hid_1_var, l_cell_init_1: cell_1_var}
            )
        mean, log_std, hid_out = outputs
        cell_out = extra_outputs[l_forward_1][0]

        f_forward = compile_function(
            inputs=[l_in.input_var, hid_1_var, cell_1_var],
            outputs=[mean, log_std, hid_out, cell_out],
        )

        self._n_hidden = n_hidden
        self._grad_clip = grad_clip
        self._f_forward = f_forward
        self._l_mean = l_mean
        self._l_log_std = l_log_std
        self._l_in = l_in
        self._l_reset = l_reset
        self._cur_hid = None
        self._cur_cell = None
        self.episode_reset()

        super(MeanStdRNNPolicy, self).__init__(mdp)
        LasagnePowered.__init__(self, [l_mean, l_log_std])
        Serializable.__init__(self, mdp)

    def _split_pdist(self, pdist):
        mean = pdist[:, :self.action_dim]
        log_std = pdist[:, self.action_dim:]
        return mean, log_std

    @overrides
    def episode_reset(self):
        self._cur_hid = np.zeros((1, self._n_hidden))
        self._cur_cell = np.zeros((1, self._n_hidden))

    @overrides
    def get_action(self, observation):
        mean, log_std, self._cur_hid, self._cur_cell = \
            [x[0] for x in
             self._f_forward(
                 observation.reshape((1, 1, -1)),
                 self._cur_hid,
                 self._cur_cell,
            )]
        rnd = np.random.randn(*mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, np.concatenate([mean, log_std])

    def get_log_prob_sym(self, obs_var, action_var, reset_var, train=False):
        # Here, we assume for now that this quantity is computed over a single
        # path
        obs_var = TT.shape_padleft(obs_var, n_ones=1)
        reset_var = TT.shape_padleft(reset_var, n_ones=1)
        means, log_stds = LH.get_output(
            [self._l_mean, self._l_log_std],
            {self._l_in: obs_var, self._l_reset: reset_var}
        )
        stdn = (action_var - means)
        stdn /= TT.exp(log_stds)
        return - TT.sum(log_stds, axis=-1) - \
            0.5*TT.sum(TT.square(stdn), axis=-1) - \
            0.5*self.action_dim*np.log(2*np.pi)

    @overrides
    def compute_entropy(self, pdist):
        _, log_std = self._split_pdist(pdist)
        return np.mean(np.sum(log_std + np.log(np.sqrt(2*np.pi*np.e)), axis=1))

    def log_extra(self, logger, paths):
        pdists = np.vstack([path["pdists"] for path in paths])
        means, log_stds = self._split_pdist(pdists)
        logger.record_tabular('AveragePolicyStd', np.mean(np.exp(log_stds)))
