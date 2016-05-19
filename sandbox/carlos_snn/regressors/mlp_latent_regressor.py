import numpy as np

from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.misc.overrides import overrides
from rllab.misc import logger

# the regressor will be choosen to be from the same distribution as the latents
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from sandbox.rocky.snn.regressors.bernoulli_mlp_regressor import BernoulliMLPRegressor

class MLPLatent_regressor(Parameterized, Serializable):

    def __init__(
            self,
            env_spec,
            policy,
            regressor_args=None,
    ):
        self.latent_dim = policy.latent_dim
        self.obs_act_dim = env_spec.observation_space.flat_dim + env_spec.action_space.flat_dim
        self.policy = policy
        Serializable.quick_init(self, locals())  # ??

        if regressor_args is None:
            regressor_args = dict()

        if policy.latent_name == 'bernoulli':
            self._regressor = BernoulliMLPRegressor(
                input_shape=(self.obs_act_dim,),
                output_dim=policy.latent_dim,
                **regressor_args
            )
        elif policy.latent_name == 'gaussian':
            self._regressor = GaussianMLPRegressor(
                input_shape=(self.obs_act_dim,),
                output_dim=policy.latent_dim,
                **regressor_args
            )
        else:
            raise NotImplementedError

    def fit(self, paths):
        observations = np.concatenate([p["observations"] for p in paths])
        actions = np.concatenate([p["actions"] for p in paths])
        obs_actions = np.concatenate([observations, actions], axis=1)
        latents = np.concatenate([p['agent_infos']["latent"] for p in paths])
        self._regressor.fit(obs_actions, latents.reshape((-1, self.latent_dim)))

    def predict(self, path):
        obs_actions = np.concatenate([path["observations"], path["actions"]], axis=1)
        return self._regressor.predict(obs_actions).flatten()

    def get_output_p(self, path):
        obs_actions = np.concatenate([path["observations"], path["actions"]], axis=1)
        if self.policy.latent_name == 'bernoulli':
            return self._regressor._f_p(obs_actions).flatten()
        elif self.policy.latent_name == 'gaussian':
            return self._regressor._f_pdists(obs_actions).flatten()

    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)

    def predict_log_likelihood(self, paths, latents):
        observations = np.concatenate([p["observations"] for p in paths])
        actions = np.concatenate([p["actions"] for p in paths])
        obs_actions = np.concatenate([observations, actions], axis=1)
        return self._regressor.predict_log_likelihood(obs_actions, latents)

    def lowb_mutual(self, paths):
        latents = np.concatenate([p["agent_infos"]["latent"] for p in paths])
        observations = np.concatenate([p["observations"] for p in paths])
        actions = np.concatenate([p["actions"] for p in paths])
        obs_actions = np.concatenate([observations, actions], axis=1)
        H_latent = self.policy.latent_dist.entropy(self.policy.latent_dist_info_vars)
        return H_latent + np.mean(self._regressor.predict_log_likelihood(obs_actions, latents))

    def log_diagnostics(self, paths):
        logger.record_tabular('LowerB_MI', self.lowb_mutual(paths))


