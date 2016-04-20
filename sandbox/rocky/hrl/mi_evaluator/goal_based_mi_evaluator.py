import numpy as np

from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.distributions.categorical import Categorical
from rllab.misc.special import to_onehot
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from rllab.regressors.categorical_mlp_regressor import CategoricalMLPRegressor
from rllab.spaces.discrete import Discrete
from rllab.spaces.box import Box
from rllab.spaces.product import Product
from sandbox.rocky.hrl import hrl_utils
from sandbox.rocky.hrl.grid_world_hrl_utils import ExactComputer
from sandbox.rocky.hrl.subgoal_policy import SubgoalPolicy


class GoalBasedMIEvaluator(LasagnePowered, Serializable):
    """
    Defines the reward bonus as the mutual information between the future state and the subgoal, given the current
    state: I(s',g|s) = H(g|s) - H(g|s,s') >= E[log(q(g|s,s')) - log(p(g|s))], where q approximates p(g|s,s')
    We train a neural network to fit either a Gaussian or a categorical distribution to p(g|s,s').
    """

    def __init__(
            self,
            env_spec,
            policy,
            component_idx=None,
            regressor_cls=None,
            regressor_args=None,

            logger_delegate=None):
        assert isinstance(policy, SubgoalPolicy)
        assert isinstance(policy.subgoal_space, (Discrete, Box))
        if component_idx is not None:
            assert isinstance(env_spec.observation_space, Product)
            assert 0 <= component_idx < len(env_spec.observation_space.components)

        Serializable.quick_init(self, locals())
        if regressor_cls is None:
            if isinstance(policy.subgoal_space, Discrete):
                regressor_cls = CategoricalMLPRegressor
            elif isinstance(policy.subgoal_space, Box):
                regressor_cls = GaussianMLPRegressor
            else:
                raise NotImplementedError
        if regressor_args is None:
            regressor_args = dict()

        self.env_spec = env_spec
        self.policy = policy
        self.component_idx = component_idx
        if component_idx is None:
            self.component_space = env_spec.observation_space
        else:
            self.component_space = env_spec.observation_space.components[component_idx]

        self.regressor = regressor_cls(
            input_shape=(env_spec.observation_space.flat_dim + self.component_space.flat_dim,),
            output_dim=policy.subgoal_space.flat_dim,
            name="(g|s,s')",
            **regressor_args
        )

        self.subgoal_space = policy.subgoal_space
        self.subgoal_interval = policy.subgoal_interval
        self.logger_delegate = logger_delegate

    def _get_relevant_data(self, paths):
        if self.component_idx is None:
            obs = np.concatenate([p["agent_infos"]["high_obs"][:-1] for p in paths])
            next_obs = np.concatenate([p["agent_infos"]["high_obs"][1:] for p in paths])
            subgoals = np.concatenate([p["agent_infos"]["subgoal"][:-1] for p in paths])
            N = obs.shape[0]
            return obs.reshape((N, -1)), next_obs.reshape((N, -1)), subgoals
        else:
            obs = np.concatenate([p["agent_infos"]["high_obs"][:-1] for p in paths])
            N = obs.shape[0]
            next_obs = np.concatenate([p["agent_infos"]["high_obs"][1:] for p in paths]).reshape((N, -1))

            obs_flat_dims = [c.flat_dim for c in self.env_spec.observation_space.components]
            slice_start = sum(obs_flat_dims[:self.component_idx])
            slice_end = slice_start + obs_flat_dims[self.component_idx]
            next_component_obs = next_obs[:, slice_start:slice_end]
            subgoals = np.concatenate([p["agent_infos"]["subgoal"][:-1] for p in paths])
            return obs.reshape((N, -1)), next_component_obs, subgoals

    def fit(self, paths):
        subsampled_paths = [hrl_utils.downsample_path(p, self.subgoal_interval) for p in paths]
        flat_obs, flat_next_obs, subgoals = self._get_relevant_data(subsampled_paths)
        xs = np.concatenate([flat_obs, flat_next_obs], axis=1)
        ys = subgoals
        self.regressor.fit(xs, ys)
        if self.logger_delegate:
            self.logger_delegate.fit(paths)

    def log_diagnostics(self, paths):
        if self.logger_delegate:
            self.logger_delegate.log_diagnostics(paths)

    def predict(self, path):
        path_length = len(path["rewards"])
        path = hrl_utils.downsample_path(path, self.subgoal_interval)
        flat_obs, flat_next_obs, subgoals = self._get_relevant_data([path])
        xs = np.concatenate([flat_obs, flat_next_obs], axis=1)
        ys = subgoals
        log_p_g_given_s_sp = self.regressor.predict_log_likelihood(xs, ys)
        high_dist_info = self.policy.high_policy.dist_info(flat_obs, subgoals)
        log_p_g_given_s = self.policy.high_policy.distribution.log_likelihood(subgoals, high_dist_info)
        bonuses = np.append(log_p_g_given_s_sp - log_p_g_given_s, 0)
        bonuses = np.tile(
            np.expand_dims(bonuses, axis=1),
            (1, self.subgoal_interval)
        ).flatten()[:path_length]
        return bonuses
