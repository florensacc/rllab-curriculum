import tensorflow as tf
import numpy as np

from sandbox.haoran.mddpg.qfunctions.nn_qfunction import NNCritic
from rllab.core.serializable import Serializable


class MixtureGaussian2DCritic(NNCritic):
    """ Q(s,a) is a 2D mixture of Gaussian in a """
    def __init__(
            self,
            scope_name,
            observation_dim,
            action_input,
            observation_input,
            weights,
            mus,
            sigmas,
            reuse=False,
            **kwargs
    ):
        Serializable.quick_init(self, locals())

        self._weights = weights
        self._mus = mus
        self._sigmas = sigmas

        super(MixtureGaussian2DCritic, self).__init__(
            scope_name=scope_name,
            observation_dim=observation_dim,
            action_dim=2,
            action_input=action_input,
            observation_input=observation_input,
            reuse=reuse,
            **kwargs
        )

    def create_network(self, action_input, observation_input):
        a = action_input
        components = []
        for w, mu, sigma in zip(self._weights, self._mus, self._sigmas):
            mu = np.reshape(mu, (1, 2))
            comp = (1./tf.sqrt(2. * np.pi * sigma**2)) * tf.exp(
                    -0.5 / sigma**2 * tf.reduce_sum(tf.square(a - mu), axis=1))

            components.append(comp)

        temp = 0.5
        output = temp * tf.log(tf.add_n(components))
        return output

    def get_weight_tied_copy(self, action_input, observation_input):
        """
        HT: basically, re-run __init__ with specified kwargs. In particular,
        the variable scope doesn't change, and self.observations_placeholder
        and NN params are reused.
        """
        return self.__class__(
            scope_name=self.scope_name,
            observation_dim=self.observation_dim,
            action_input=action_input,
            observation_input=observation_input,
            weights=self._weights.copy(),
            mus=self._mus.copy(),
            sigmas=self._sigmas.copy(),
            reuse=True,
        )
