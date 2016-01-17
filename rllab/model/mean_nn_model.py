import theano
import theano.tensor as TT
import lasagne.layers as L
import lasagne.init as LI
import lasagne.nonlinearities as NL
import numpy as np
from collections import OrderedDict
from rllab.model.base import Model
from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.misc import autoargs
from rllab.misc.special import normalize_updates
from rllab.misc.ext import new_tensor
from rllab.misc.overrides import overrides


class MeanNNModel(Model, LasagnePowered, Serializable):

    """
    A deterministic model, parameterized by a neural network.
    """

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
        action_var = TT.matrix(
            'action',
            dtype=mdp.action_dtype
        )
        obs_layer = L.InputLayer(shape=(None,) + mdp.observation_shape,
                                 input_var=obs_var)
        action_layer = L.InputLayer(shape=(None, mdp.action_dim),
                                    input_var=action_var)

        next_obs_layer = self._create_network(
            mdp, obs_layer, action_layer, mdp.observation_shape[0])

        reward_layer = self._create_network(
            mdp, obs_layer, action_layer, 1)

        assert len(mdp.observation_shape) == 1

        if self.normalize:
            self._next_obs_mean = theano.shared(
                np.zeros((1,) + mdp.observation_shape, dtype='float32'),
                broadcastable=(True,) + (False,) * len(mdp.observation_shape)
            )
            self._next_obs_std = theano.shared(
                np.ones((1,) + mdp.observation_shape, dtype='float32'),
                broadcastable=(True,) + (False,) * len(mdp.observation_shape)
            )
            self._reward_mean = theano.shared(
                np.zeros(1, dtype='float32'),
                broadcastable=(True,)
            )
            self._reward_std = theano.shared(
                np.ones(1, dtype='float32'),
                broadcastable=(True,)
            )

        self._in_compute_normalize = False

        self._obs_layer = obs_layer
        self._action_layer = action_layer
        self._next_obs_layer = next_obs_layer
        self._reward_layer = reward_layer

        Model.__init__(self, mdp)
        LasagnePowered.__init__(self, [next_obs_layer, reward_layer])
        Serializable.__init__(self, mdp=mdp)

    def _create_network(self, mdp, obs_layer, action_layer, output_units):
        hidden_layer = L.DenseLayer(
            obs_layer,
            num_units=100,
            W=LI.HeUniform(),
            b=LI.Constant(0.),
            nonlinearity=NL.rectify,
            name="h0"
        )
        hidden_layer = L.ConcatLayer([hidden_layer, action_layer])
        hidden_layer = L.DenseLayer(
            hidden_layer,
            num_units=100,
            W=LI.HeUniform(),
            b=LI.Constant(0.),
            nonlinearity=NL.rectify,
            name="h1"
        )

        output_layer = L.DenseLayer(
            hidden_layer,
            num_units=output_units,
            W=LI.HeUniform(),
            b=LI.Constant(0.),
            nonlinearity=None,
            name="output"
        )

        return output_layer

    @overrides
    def predict_obs_sym(self, obs_var, action_var, train=False):
        next_obs = L.get_output(
            self._next_obs_layer,
            {self._obs_layer: obs_var, self._action_layer: action_var}
        )
        if self.normalize:
            next_obs = next_obs * self._next_obs_std + self._next_obs_mean
        return next_obs

    @overrides
    def predict_reward_sym(self, obs_var, action_var, train=False):
        rewards = L.get_output(
            self._reward_layer,
            {self._obs_layer: obs_var, self._action_layer: action_var}
        )
        rewards = TT.reshape(rewards, (-1,))
        if self.normalize:
            rewards = rewards * self._reward_std + self._reward_mean
        return rewards

    @overrides
    def obs_regression_obj(self, obs_var, action_var, next_obs_var,
                           train=False):
        predicted = self.predict_obs_sym(obs_var, action_var, train=train)
        diff = predicted - next_obs_var
        if self.normalize:
            diff = diff / self._next_obs_std
        return TT.mean(TT.square(diff))

    def next_obs_normalize_updates(self, next_obs_var):
        if not self.normalize:
            return OrderedDict([])
        mean = TT.mean(next_obs_var, axis=0, keepdims=True)
        std = TT.std(next_obs_var, axis=0, keepdims=True)
        return normalize_updates(
            old_mean=self._next_obs_mean,
            old_std=self._next_obs_std,
            new_mean=self._running_sum(mean, self._next_obs_mean),
            new_std=self._running_sum(std, self._next_obs_std),
            old_W=self._next_obs_layer.W,
            old_b=self._next_obs_layer.b,
        )

    def reward_normalize_updates(self, reward_var):
        if not self.normalize:
            return {}
        mean = TT.mean(reward_var, axis=0, keepdims=True)
        std = TT.std(reward_var, axis=0, keepdims=True)
        return normalize_updates(
            old_mean=self._reward_mean,
            old_std=self._reward_std,
            new_mean=self._running_sum(mean, self._reward_mean),
            new_std=self._running_sum(std, self._reward_std),
            old_W=self._reward_layer.W,
            old_b=self._reward_layer.b,
        )

    def _running_sum(self, new, old):
        return self.normalize_alpha * new + (1 - self.normalize_alpha) * old

    @overrides
    def reward_regression_obj(self, obs_var, action_var, reward_var,
                              train=False):
        predicted = self.predict_reward_sym(obs_var, action_var, train=train)
        diff = predicted - reward_var
        if self.normalize:
            diff = diff / self._reward_std
        return TT.mean(TT.square(diff))
