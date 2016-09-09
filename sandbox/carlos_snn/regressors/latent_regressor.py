import numpy as np

from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.misc.overrides import overrides
from rllab.misc import logger

# the regressor will be choosen to be from the same distribution as the latents
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from rllab.regressors.categorical_mlp_regressor import CategoricalMLPRegressor
from sandbox.carlos_snn.regressors.categorical_recurrent_regressor import CategoricalRecurrentRegressor
from sandbox.carlos_snn.regressors.bernoulli_mlp_regressor import BernoulliMLPRegressor
from sandbox.carlos_snn.regressors.bernoulli_recurrent_regressor import BernoulliRecurrentRegressor

from rllab.optimizers.first_order_optimizer import FirstOrderOptimizer


class Latent_regressor(Parameterized, Serializable):
    def __init__(
            self,
            env_spec,
            policy,
            recurrent=False,
            predict_all=True,
            obs_regressed='all',
            act_regressed='all',
            use_only_sign=False,
            noisify_traj_coef=0,
            optimizer=None,  # this defaults to LBFGS
            regressor_args=None,  # here goes all args straight to the regressor: hidden_sizes, TR, step_size....
    ):
        """
        :param env_spec:
        :param policy:
        :param recurrent:
        :param predict_all: this is only for the recurrent case, to use all hidden states as predictions
        :param obs_regressed: list of index of the obs variables used to fit the regressor. default string 'all'
        :param act_regressed: list of index of the act variables used to fit the regressor. default string 'all'
        :param regressor_args:
        """
        self.env_spec = env_spec
        self.policy = policy
        self.latent_dim = policy.latent_dim
        self.recurrent = recurrent
        self.predict_all = predict_all
        self.use_only_sign = use_only_sign
        self.noisify_traj_coef = noisify_traj_coef
        self.regressor_args = regressor_args
        # decide what obs variables will be regressed upon
        if obs_regressed == 'all':
            self.obs_regressed = list(range(env_spec.observation_space.flat_dim))
        else:
            self.obs_regressed = obs_regressed
        # decide what action variables will be regressed upon
        if act_regressed == 'all':
            self.act_regressed = list(range(env_spec.action_space.flat_dim))
        else:
            self.act_regressed = act_regressed
        # shape the input dimension of the NN for the above decisions.
        self.obs_act_dim = len(self.obs_regressed) + len(self.act_regressed)

        Serializable.quick_init(self, locals())  # ??

        if regressor_args is None:
            regressor_args = dict()

        if optimizer == 'first_order':
            self.optimizer = FirstOrderOptimizer(
                max_epochs=10,  # both of these are to match Rocky's 10
                batch_size=128,
            )
        elif optimizer is None:
            self.optimizer = None
        else:
            raise NotImplementedError

        if policy.latent_name == 'bernoulli':
            if self.recurrent:
                self._regressor = BernoulliRecurrentRegressor(
                    input_shape=(self.obs_act_dim,),
                    output_dim=policy.latent_dim,
                    optimizer=self.optimizer,
                    predict_all=self.predict_all,
                    **regressor_args
                )
            else:
                self._regressor = BernoulliMLPRegressor(
                    input_shape=(self.obs_act_dim,),
                    output_dim=policy.latent_dim,
                    optimizer=self.optimizer,
                    **regressor_args
                )
        elif policy.latent_name == 'categorical':
            if self.recurrent:
                print('setting a recurrent categorical regressor')
                self._regressor = CategoricalRecurrentRegressor(  # not implemented
                    input_shape=(self.obs_act_dim,),
                    output_dim=policy.latent_dim,
                    optimizer=self.optimizer,
                    # predict_all=self.predict_all,
                    **regressor_args
                )
            else:
                print('setting a MLP categorical regressor')
                self._regressor = CategoricalMLPRegressor(
                    input_shape=(self.obs_act_dim,),
                    output_dim=policy.latent_dim,
                    optimizer=self.optimizer,
                    **regressor_args
                )
        elif policy.latent_name == 'normal':
            self._regressor = GaussianMLPRegressor(
                input_shape=(self.obs_act_dim,),
                output_dim=policy.latent_dim,
                optimizer=self.optimizer,
                **regressor_args
            )
        else:
            raise NotImplementedError

    def fit(self, paths):
        logger.log('fitting the regressor...')
        if self.recurrent:
            observations = np.array([p["observations"][:, self.obs_regressed] for p in paths])
            # print 'the obs shape is: ', np.shape(observations)
            actions = np.array([p["actions"][:, self.act_regressed] for p in paths])
            # print 'the actions shape is; ', np.shape(actions)
            obs_actions = np.concatenate([observations, actions], axis=2)
            if self.noisify_traj_coef:
                obs_actions += np.random.normal(loc=0.0,
                                                scale=float(np.mean(np.abs(obs_actions))) * self.noisify_traj_coef,
                                                size=np.shape(obs_actions))
            latents = np.array([p['agent_infos']['latents'] for p in paths])
            self._regressor.fit(obs_actions, latents)  # the input shapes are (traj, time, dim)
        else:
            observations = np.concatenate([p["observations"][:, self.obs_regressed] for p in paths])
            actions = np.concatenate([p["actions"][:, self.act_regressed] for p in paths])
            obs_actions = np.concatenate([observations, actions], axis=1)
            latents = np.concatenate([p['agent_infos']["latents"] for p in paths])
            if self.noisify_traj_coef:
                obs_actions += np.random.normal(loc=0.0,
                                                scale=float(np.mean(np.abs(obs_actions))) * self.noisify_traj_coef,
                                                size=np.shape(obs_actions))
            self._regressor.fit(obs_actions, latents.reshape((-1, self.latent_dim)))  # why reshape??
        logger.log('done fitting the regressor')

    def predict(self, path):
        if self.recurrent:
            obs_actions = [np.concatenate([path["observations"][:, self.obs_regressed],
                                           path["actions"][:, self.act_regressed]],
                                          axis=1)]  # is this the same??
        else:
            obs_actions = np.concatenate([path["observations"][:, self.obs_regressed],
                                          path["actions"][:, self.act_regressed]], axis=1)
        if self.noisify_traj_coef:
            obs_actions += np.random.normal(loc=0.0, scale=float(np.mean(np.abs(obs_actions))) * self.noisify_traj_coef,
                                            size=np.shape(obs_actions))
        if self.use_only_sign:
            obs_actions = np.sign(obs_actions)
        return self._regressor.predict(obs_actions).flatten()

    def get_output_p(self, path):  # this gives the p_dist for every step: the latent posterior wrt obs_act
        if self.recurrent:
            obs_actions = [np.concatenate([path["observations"][:, self.obs_regressed],
                                           path["actions"][:, self.act_regressed]],
                                          axis=1)]  # is this the same??
        else:
            obs_actions = np.concatenate([path["observations"][:, self.obs_regressed],
                                          path["actions"][:, self.act_regressed]], axis=1)
        if self.noisify_traj_coef:
            obs_actions += np.random.normal(loc=0.0, scale=float(np.mean(np.abs(obs_actions))) * self.noisify_traj_coef,
                                            size=np.shape(obs_actions))
        if self.use_only_sign:
            obs_actions = np.sign(obs_actions)
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
            observations = np.array([p["observations"][:, self.obs_regressed] for p in paths])
            actions = np.array([p["actions"][:, self.act_regressed] for p in paths])
            obs_actions = np.concatenate([observations, actions], axis=2)  # latents must match first 2dim: (batch,time)
            # print 'CF the obs are:', observations, 'CF the act are: ', actions, 'CF the combined are: ', obs_actions
        else:
            observations = np.concatenate([p["observations"][:, self.obs_regressed] for p in paths])
            actions = np.concatenate([p["actions"][:, self.act_regressed] for p in paths])
            obs_actions = np.concatenate([observations, actions], axis=1)
            latents = np.concatenate(latents, axis=0)
        if self.noisify_traj_coef:
            noise = np.random.multivariate_normal(mean=np.zeros_like(np.mean(obs_actions, axis=0)),
                                                         cov=np.diag(np.mean(np.abs(obs_actions),
                                                                     axis=0) * self.noisify_traj_coef),
                                                         size=np.shape(obs_actions)[0])
            obs_actions += noise
        if self.use_only_sign:
            obs_actions = np.sign(obs_actions)
        return self._regressor.predict_log_likelihood(obs_actions, latents)  # see difference with fit above...

    def lowb_mutual(self, paths, times=(0, None)):
        if self.recurrent:
            observations = np.array([p["observations"][times[0]:times[1], self.obs_regressed] for p in paths])
            actions = np.array([p["actions"][times[0]:times[1], self.act_regressed] for p in paths])
            obs_actions = np.concatenate([observations, actions], axis=2)
            latents = np.array([p['agent_infos']['latents'][times[0]:times[1]] for p in paths])
        else:
            observations = np.concatenate([p["observations"][times[0]:times[1], self.obs_regressed] for p in paths])
            actions = np.concatenate([p["actions"][times[0]:times[1], self.act_regressed] for p in paths])
            obs_actions = np.concatenate([observations, actions], axis=1)
            latents = np.concatenate([p['agent_infos']["latents"][times[0]:times[1]] for p in paths])
        if self.noisify_traj_coef:
            obs_actions += np.random.multivariate_normal(mean=np.zeros_like(np.mean(obs_actions,axis=0)),
                                                         cov=np.diag(np.mean(np.abs(obs_actions),
                                                                     axis=0) * self.noisify_traj_coef),
                                                         size=np.shape(obs_actions)[0])
        if self.use_only_sign:
            obs_actions = np.sign(obs_actions)
        H_latent = self.policy.latent_dist.entropy(self.policy.latent_dist_info)  # sum of entropies latents in
        # one timestep (assumes iid)
        # print 'the latent entropy is: ', H_latent
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
        logger.record_tabular(self._regressor._name + 'LowerB_MI', self.lowb_mutual(paths))
        logger.record_tabular(self._regressor._name + 'LowerB_MI_5first', self.lowb_mutual(paths, times=(0, 5)))
        logger.record_tabular(self._regressor._name + 'LowerB_MI_5last', self.lowb_mutual(paths, times=(-5, None)))
