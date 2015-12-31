import theano
import numpy as np
import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne.init as LI
import theano.tensor as TT
from rllab.vf.base import ValueFunction
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.misc import autoargs
from rllab.misc.ext import new_tensor
from rllab.misc.overrides import overrides


class NNValueFunction(ValueFunction, LasagnePowered, Serializable):

    @autoargs.arg('normalize', type=bool,
                  help='Whether to normalize the output.')
    @autoargs.arg('normalize_alpha', type=float,
                  help='Coefficient for the running mean and std.')
    def __init__(
            self,
            mdp,
            normalize=True,
            normalize_alpha=0.1):

        self.normalize = normalize
        self.normalize_alpha = normalize_alpha

        obs_var = new_tensor(
            'obs',
            ndim=1+len(mdp.observation_shape),
            dtype=mdp.observation_dtype
        )

        obs_layer = L.InputLayer(shape=(None,) + mdp.observation_shape,
                                 input_var=obs_var)

        h0_layer = L.DenseLayer(
            obs_layer,
            num_units=200,
            W=LI.HeUniform(),
            b=LI.Constant(0.),
            nonlinearity=NL.rectify,
            name="h0"
        )

        h1_layer = L.DenseLayer(
            h0_layer,
            num_units=100,
            W=LI.HeUniform(),
            b=LI.Constant(0.),
            nonlinearity=NL.rectify,
            name="h1"
        )

        output_layer = L.DenseLayer(
            h1_layer,
            num_units=1,
            W=LI.HeUniform(),
            b=LI.Constant(0.),
            nonlinearity=NL.rectify,
            name="output"
        )

        self._output_layer = output_layer
        self._obs_layer = obs_layer

        if self.normalize:
            self._val_mean = theano.shared(np.zeros(1), broadcastable=(True,))
            self._val_std = theano.shared(np.ones(1), broadcastable=(True,))

        ValueFunction.__init__(self, mdp)
        LasagnePowered.__init__(self, [output_layer])
        Serializable.__init__(self, mdp)

    @overrides
    def get_val_sym(self, obs_var, train=False):
        vals = L.get_output(self._output_layer, obs_var)
        vals = TT.reshape(vals, (-1,))
        if self.normalize:
            vals = vals * self._val_std + self._val_mean
        return vals

    def _running_sum(self, new, old):
        return self.normalize_alpha * new + (1 - self.normalize_alpha) * old

    def normalize_updates(self, ys):
        mean = TT.mean(ys, axis=0, keepdims=True)
        std = TT.std(ys, axis=0, keepdims=True)
        return {
            self._val_mean: self._running_sum(mean, self._val_mean),
            self._val_std: self._running_sum(std, self._val_std),
        }

    def normalize_sym(self, vals):
        if self.normalize:
            return (vals - self._val_mean) / self._val_std
        return vals
