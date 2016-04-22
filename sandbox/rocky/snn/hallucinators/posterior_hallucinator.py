from __future__ import print_function
from __future__ import absolute_import
from rllab.core.serializable import Serializable
from rllab.spaces.product import Product
from rllab.spaces.box import Box
from rllab.regressors.product_regressor import ProductRegressor
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from sandbox.rocky.snn.regressors.bernoulli_mlp_regressor import BernoulliMLPRegressor
from sandbox.rocky.snn.core.lasagne_layers import IndependentGaussianLayer, IndependentBernoulliLayer, GaussianLayer, \
    BernoulliLayer
import numpy as np


class PosteriorHallucinator(Serializable):
    """
    Hallucinate additional samples by sampling from the posterior p(h|s,a). Since the exact posterior is in general
    intractable, we resort to fitting an approximate posterior.

    We use a mean-field approximation to the posterior. Specifically we assume a factorized distribution.
    """

    def __init__(self, env_spec, policy, n_hallucinate_samples=5, fit_before=True, regressor_args=None):
        """
        :param policy: policy to resample from
        :param n_hallucinate_samples: number of hallucinated samples for each real sample
        :param fit_before: whether to fit the posterior before or after hallucination in each iteration
        :return:
        """
        Serializable.quick_init(self, locals())
        self.policy = policy
        self.n_hallucinate_samples = n_hallucinate_samples
        self.fit_before = fit_before

        if regressor_args is None:
            regressor_args = dict()
        regressor_input_space = Product(env_spec.observation_space, env_spec.action_space)
        regressor_input_shape = (regressor_input_space.flat_dim,)
        regressor_output_spaces = []
        component_regressors = []
        for latent_layer in self.policy.latent_layers:
            if isinstance(latent_layer, (IndependentBernoulliLayer, BernoulliLayer)):
                regressor_cls = BernoulliMLPRegressor
            elif isinstance(latent_layer, (IndependentGaussianLayer, GaussianLayer)):
                regressor_cls = GaussianMLPRegressor
            else:
                raise NotImplementedError
            component_regressors.append(
                regressor_cls(
                    input_shape=regressor_input_shape,
                    output_dim=latent_layer.num_units,
                    **regressor_args
                )
            )
            regressor_output_spaces.append(Box(low=-np.inf, high=np.inf, shape=(latent_layer.num_units,)))
        regressor_output_space = Product(regressor_output_spaces)

        self.regressor = ProductRegressor(
            component_regressors
        )
        self.regressor_output_space = regressor_output_space

    def fit_posterior(self, samples_data):
        observations = samples_data["observations"]
        actions = samples_data["actions"]
        agent_infos = samples_data["agent_infos"]
        xs = np.concatenate([observations, actions], axis=1)
        ys = []
        for idx in xrange(len(self.policy.latent_layers)):
            ys.append(agent_infos["latent_%d" % idx])
        ys = np.concatenate(ys, axis=1)
        self.regressor.fit(xs, ys)

    def hallucinate_from_posterior(self, samples_data):
        observations = samples_data["observations"]
        actions = samples_data["actions"]
        agent_infos = samples_data["agent_infos"]
        xs = np.concatenate([observations, actions], axis=1)
        dist = self.policy.distribution
        old_logli = dist.log_likelihood(actions, agent_infos)
        for _ in xrange(self.n_hallucinate_samples):
            latents = self.regressor.sample_predict(xs)
            latent_logli = self.regressor.predict_log_likelihood(xs, latents)
            splitted_latents = self.regressor_output_space.unflatten_n(latents)
            latents_dict = dict()
            # need to split the latent variables
            for idx, latent_i in enumerate(splitted_latents):
                latents_dict["latent_%d" % idx] = latent_i
            import ipdb; ipdb.set_trace()
            new_dist_info = self.policy.dist_info(xs, latents_dict)

        # TODO
        pass

    def hallucinate(self, samples_data):
        if self.fit_before:
            self.fit_posterior(samples_data)
            return self.hallucinate_from_posterior(samples_data)
        else:
            ret = self.hallucinate_from_posterior(samples_data)
            self.fit_posterior(samples_data)
            return ret
