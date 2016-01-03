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


def lstm_cloneable_args(lstm_layer):
    return dict(
        num_units=lstm_layer.num_units,
        grad_clipping=lstm_layer.grad_clipping,
        nonlinearity=lstm_layer.nonlinearity,
        ingate=LR.Gate(
            W_in=lstm_layer.W_in_to_ingate,
            W_hid=lstm_layer.W_hid_to_ingate,
            W_cell=getattr(lstm_layer, "W_cell_to_ingate", None),
            b=lstm_layer.b_ingate
        ),
        forgetgate=LR.Gate(
            W_in=lstm_layer.W_in_to_forgetgate,
            W_hid=lstm_layer.W_hid_to_forgetgate,
            W_cell=getattr(lstm_layer, "W_cell_to_forgetgate", None),
            b=lstm_layer.b_forgetgate
        ),
        cell=LR.Gate(
            W_in=lstm_layer.W_in_to_cell,
            W_hid=lstm_layer.W_hid_to_cell,
            b=lstm_layer.b_cell,
            nonlinearity=lstm_layer.nonlinearity_cell,
        ),
        outgate=LR.Gate(
            W_in=lstm_layer.W_in_to_outgate,
            W_hid=lstm_layer.W_hid_to_outgate,
            W_cell=getattr(lstm_layer, "W_cell_to_outgate", None),
            b=lstm_layer.b_outgate
        ),
        hid_init=lstm_layer.hid_init,
        cell_init=lstm_layer.cell_init
    )


class MeanStdRNNPolicy(StochasticPolicy, LasagnePowered, Serializable):

    def __init__(self, mdp):
        # create network

        n_hidden = 10  # 32
        grad_clip = 100

        # forget_gate = LR.Gate(b=lasagne.init.Constant(5.0))

        l_in = L.InputLayer(shape=(None, None, mdp.observation_shape[0]))

        hid_init_1 = theano.shared(
            np.zeros((1, n_hidden), dtype=theano.config.floatX))
        cell_init_1 = theano.shared(
            np.zeros((1, n_hidden), dtype=theano.config.floatX))
        l_hid_init_1 = L.InputLayer(
            input_var=hid_init_1, shape=(1, n_hidden))
        l_cell_init_1 = L.InputLayer(
            input_var=cell_init_1, shape=(1, n_hidden))

        l_forward_1 = LR.LSTMLayer(
            l_in,
            hid_init=l_hid_init_1,
            cell_init=l_cell_init_1,
            num_units=n_hidden,
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

        outputs, extra_outputs = \
            LH.get_output_with_extra([l_mean, l_log_std, l_forward_1])
        mean, log_std, hid_out = outputs
        cell_out = extra_outputs[l_forward_1][0]

        hid_1_var = TT.matrix('hid_1', dtype=hid_out.dtype)
        cell_1_var = TT.matrix('cell_1', dtype=hid_out.dtype)

        cont_outputs, cont_extra_outputs = \
            LH.get_output_with_extra(
                [l_mean, l_log_std, l_forward_1],
                {l_hid_init_1: hid_1_var, l_cell_init_1: cell_1_var}
            )
        cont_mean, cont_log_std, cont_hid_out = cont_outputs
        cont_cell_out = cont_extra_outputs[l_forward_1][0]

        f_initial_forward = compile_function(
            inputs=[l_in.input_var],
            outputs=[mean, log_std, hid_out, cell_out],
        )

        f_cont_forward = compile_function(
            inputs=[l_in.input_var, hid_1_var, cell_1_var],
            outputs=[cont_mean, cont_log_std, cont_hid_out, cont_cell_out],
        )

        self._f_initial_forward = f_initial_forward
        self._f_cont_forward = f_cont_forward
        self._l_mean = l_mean
        self._l_log_std = l_log_std
        self._l_in = l_in
        self._obs_history = []
        self._cur_hid = None
        self._cur_cell = None

        super(MeanStdRNNPolicy, self).__init__(mdp)
        LasagnePowered.__init__(self, [l_mean, l_log_std])
        Serializable.__init__(self, mdp)

        print self.params

    def _split_pdist(self, pdist):
        mean = pdist[:, :self.action_dim]
        log_std = pdist[:, self.action_dim:]
        return mean, log_std

    @overrides
    def episode_reset(self):
        self._cur_hid = None
        self._cur_cell = None

    @overrides
    def get_action(self, observation):
        if self._cur_hid is None:
            mean, log_std, hid, cell = \
                [x[0] for x in
                 self._f_initial_forward(observation.reshape((1, 1, -1)))]
            self._cur_hid = hid
            self._cur_cell = cell
        else:
            mean, log_std, hid, cell = \
                [x[0] for x in
                 self._f_cont_forward(
                     observation.reshape((1, 1, -1)),
                     self._cur_hid,
                     self._cur_cell,
                )]
            self._cur_hid = hid
            self._cur_cell = cell
        rnd = np.random.randn(*mean.shape)
        action = rnd * np.exp(log_std) + mean
        return action, np.concatenate([mean, log_std])

    def get_log_prob_sym(self, obs_var, action_var, train=False):
        # Here, we assume for now that this quantity is computed over a single
        # path
        obs_var = TT.shape_padleft(obs_var, n_ones=1)
        means, log_stds = LH.get_output(
            [self._l_mean, self._l_log_std],
            {self._l_in: obs_var}
        )
        print "means", means.ndim
        print "log_stds", log_stds.ndim
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
