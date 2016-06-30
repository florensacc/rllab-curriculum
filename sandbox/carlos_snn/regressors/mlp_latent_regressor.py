import numpy as np

from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.misc.overrides import overrides
from rllab.misc import logger

# the regressor will be choosen to be from the same distribution as the latents
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from sandbox.carlos_snn.regressors.bernoulli_mlp_regressor import BernoulliMLPRegressor
from sandbox.carlos_snn.regressors.bernoulli_recurrent_regressor import BernoulliRecurrentRegressor

class MLPLatent_regressor(Parameterized, Serializable):

    def __init__(
            self,
            env_spec,
            policy,
            recurrent=False,
            regressor_args=None,
    ):
        self.latent_dim = policy.latent_dim
        self.obs_act_dim = env_spec.observation_space.flat_dim + env_spec.action_space.flat_dim
        self.policy = policy
        self.recurrent = recurrent
        Serializable.quick_init(self, locals())  # ??

        if regressor_args is None:
            regressor_args = dict()

        if policy.latent_name == 'bernoulli':
            if recurrent:
                self._regressor = BernoulliRecurrentRegressor(
                    input_shape=(self.obs_act_dim,),
                    output_dim=policy.latent_dim,
                    **regressor_args
                )
            else:
                self._regressor = BernoulliMLPRegressor(
                    input_shape=(self.obs_act_dim,),
                    output_dim=policy.latent_dim,
                    **regressor_args
                )
        elif policy.latent_name == 'normal':
            self._regressor = GaussianMLPRegressor(
                input_shape=(self.obs_act_dim,),
                output_dim=policy.latent_dim,
                **regressor_args
            )
        else:
            raise NotImplementedError

    def fit(self, paths):
        if self.recurrent:
            observations = np.array([p["observations"] for p in paths])
            # print 'the obs shape is: ', np.shape(observations)
            actions = np.array([p["actions"] for p in paths])
            # print 'the actions shape is; ', np.shape(actions)
            obs_actions = np.concatenate([observations, actions], axis=2)
            latents = np.array([p['agent_infos']['latents'] for p in paths])
            self._regressor.fit(obs_actions, latents) # why reshape??
        else:
            observations = np.concatenate([p["observations"] for p in paths])
            actions = np.concatenate([p["actions"] for p in paths])
            obs_actions = np.concatenate([observations, actions], axis=1)
            latents = np.concatenate([p['agent_infos']["latents"] for p in paths])
            self._regressor.fit(obs_actions, latents.reshape((-1, self.latent_dim))) # why reshape??

    def predict(self, path):
        if self.recurrent:
            obs_actions = [np.concatenate([path["observations"], path["actions"]], axis=1)] # is this the same??
        else:
            obs_actions = np.concatenate([path["observations"], path["actions"]], axis=1)
        return self._regressor.predict(obs_actions).flatten()

    def get_output_p(self, path): # this gives the p_dist for every step: the latent posterior wrt obs_act
        if self.recurrent:
            obs_actions = [np.concatenate([path["observations"], path["actions"]], axis=1)] # is this the same??
        else:
            obs_actions = np.concatenate([path["observations"], path["actions"]], axis=1)
        if self.policy.latent_name == 'bernoulli':
            return self._regressor._f_p(obs_actions).flatten()
        elif self.policy.latent_name == 'normal':
            return self._regressor._f_pdists(obs_actions).flatten()

    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)

    def predict_log_likelihood(self, paths, latents):
        if self.recurrent:
            observations = np.array([p["observations"] for p in paths])
            actions = np.array([p["actions"] for p in paths])
            obs_actions = np.concatenate([observations, actions], axis=2)
        else:
            observations = np.concatenate([p["observations"] for p in paths])
            actions = np.concatenate([p["actions"] for p in paths])
            obs_actions = np.concatenate([observations, actions], axis=1)
            latents = np.concatenate(latents, axis=0)
        return self._regressor.predict_log_likelihood(obs_actions, latents)  # see difference with fit above...

    def lowb_mutual(self, paths):
        if self.recurrent:
            observations = np.array([p["observations"] for p in paths])
            actions = np.array([p["actions"] for p in paths])
            obs_actions = np.concatenate([observations, actions], axis=2)
            latents = np.array([p['agent_infos']['latents'] for p in paths])
        else:
            observations = np.concatenate([p["observations"] for p in paths])
            actions = np.concatenate([p["actions"] for p in paths])
            obs_actions = np.concatenate([observations, actions], axis=1)
            latents = np.concatenate([p['agent_infos']["latents"] for p in paths])
        H_latent = self.policy.latent_dist.entropy(self.policy.latent_dist_info)  # sum of entropies latents in
                                                                                  # one timestep (assumes iid)
        print 'the latent entropy is: ', H_latent
        # for one Bernoulli it will be
        # 1Ber, the latent entropy is:  0.69314716056
        # 2Ber, the latent entropy is:  1.38629432112
        # 5Ber, the latent entropy is:  3.4657358028

        return H_latent + np.mean(self._regressor.predict_log_likelihood(obs_actions, latents))

        # log_likelihoods = []
        # for path in paths:
        #     obs_actions = np.concatenate([path["observations"],path["actions"]], axis=1)
        #     log_likelihoods.append(self._regressor.predict_log_likelihood(
        #                                                         obs_actions, path["agent_infos"]["latents"]))
        # H_latent = self.policy.latent_dist.entropy(self.policy.latent_dist_info) # sum of entropies (assumes iid)
        # lowb = np.mean(np.sum(log_likelihoods, axis=1))
        # return H_latent + lowb

    def log_diagnostics(self, paths):
        logger.record_tabular('LowerB_MI', self.lowb_mutual(paths))


