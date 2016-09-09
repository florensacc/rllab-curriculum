import numpy as np

from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.misc.special import to_onehot
from rllab.regressors.categorical_mlp_regressor import CategoricalMLPRegressor
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
from sandbox.rocky.hrl import hrl_utils
from sandbox.rocky.hrl.policies.subgoal_policy import SubgoalPolicy


def regressor_cls_from_space(space):
    if isinstance(space, Discrete):
        return CategoricalMLPRegressor
    elif isinstance(space, Box):
        return GaussianMLPRegressor
    else:
        raise NotImplementedError


class StateBasedMIEvaluator(LasagnePowered, Serializable):
    """
    Defines the reward bonus as the mutual information between the future state and the subgoal, given the current
    state: I(s',g|s) = H(s'|s) - H(s'|g,s) = E[log(p(s'|g,s)) - log(p(s'|s))]
    We train a neural network to fit a Gaussian to p(s'|g,s). Since we only have a discrete set of goals,
    we can marginalize over them to get p(s'|s):
    p(s'|s) = sum_g p(g|s)p(s'|g,s)
    """

    def __init__(
            self,
            env,
            policy,
            # how many samples to estimate marginal p(s'|s,g)
            n_subgoal_samples=10,
            use_state_regressor=False,
            state_regressor_cls=None,
            state_regressor_args=None,
            component_idx=None,
            regressor_cls=None,
            regressor_args=None,
            logger_delegate=None,
            logger_delegates=None):
        assert isinstance(policy, SubgoalPolicy)
        # if isinstance(policy.subgoal_space, Discrete):
        #     assert isinstance(policy.high_policy.distribution, Categorical)
        # elif isinstance(policy.subgoal_space, Box):
        # assert isinstance(policy.subgoal_space, Discrete)
        # assert isinstance(policy.high_policy.distribution, Categorical)
        # assert isinstance(policy.low_policy.distribution, Categorical)

        env_spec = env.spec

        if component_idx is not None:
            assert isinstance(env_spec.observation_space, Product)
            assert 0 <= component_idx < len(env_spec.observation_space.components)

        Serializable.quick_init(self, locals())

        self.component_idx = component_idx
        if component_idx is None:
            self.component_space = env_spec.observation_space
        else:
            self.component_space = env_spec.observation_space.components[component_idx]

        if regressor_cls is None:
            regressor_cls = regressor_cls_from_space(self.component_space)
        if regressor_args is None:
            regressor_args = dict()

        self.regressor = regressor_cls(
            input_shape=(env_spec.observation_space.flat_dim + policy.subgoal_space.flat_dim,),
            output_dim=env_spec.observation_space.flat_dim,
            name="(s'|g,s)",
            **regressor_args
        )
        # self.use_state_regressor = use_state_regressor
        if use_state_regressor:
            if state_regressor_cls is None:
                state_regressor_cls = regressor_cls_from_space(self.component_space)
            if state_regressor_args is None:
                state_regressor_args = dict()
            self.state_regressor = state_regressor_cls(
                input_shape=(env_spec.observation_space.flat_dim,),
                output_dim=env_spec.observation_space.flat_dim,
                name="(s'|s)",
                **state_regressor_args
            )
        else:
            self.state_regressor = None
        self.env_spec = env_spec
        self.policy = policy
        self.subgoal_space = policy.subgoal_space
        self.subgoal_interval = policy.subgoal_interval
        if logger_delegates is None:
            if logger_delegate is not None:
                logger_delegates = [logger_delegate]
            else:
                logger_delegates = []
        self.logger_delegates = logger_delegates
        self.n_subgoal_samples = n_subgoal_samples

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
        downsampled_paths = [hrl_utils.downsample_path(p, self.subgoal_interval) for p in paths]
        flat_obs, flat_next_obs, subgoals = self._get_relevant_data(downsampled_paths)
        xs = np.concatenate([flat_obs, subgoals], axis=1)
        ys = flat_next_obs
        self.regressor.fit(xs, ys)
        if self.state_regressor is not None:
            self.state_regressor.fit(flat_obs, flat_next_obs)
        for d in self.logger_delegates:
            d.fit(paths)

    def log_diagnostics(self, paths):
        for d in self.logger_delegates:
            d.log_diagnostics(paths)

    def predict(self, path):
        path_length = len(path["rewards"])
        # No MI reward bonus if trajectory terminated too fast
        if path_length <= self.subgoal_interval:
            return np.zeros_like(path["rewards"])
        downsampled_path = hrl_utils.downsample_path(path, self.subgoal_interval)
        flat_obs, flat_next_obs, subgoals = self._get_relevant_data([downsampled_path])
        N = flat_obs.shape[0]
        xs = np.concatenate([flat_obs, subgoals], axis=1)
        ys = flat_next_obs
        log_p_sprime_given_g_s = self.regressor.predict_log_likelihood(xs, ys)
        if self.state_regressor is not None:
            log_p_sprime_given_s = self.state_regressor.predict_log_likelihood(flat_obs, flat_next_obs)
        else:
            # need a more numerically stable way
            log_components = []
            if isinstance(self.subgoal_space, Discrete):
                high_probs = downsampled_path["agent_infos"]["high"]["prob"]
                n_subgoals = self.subgoal_space.n
                for goal in range(n_subgoals):
                    goal_mat = np.tile(to_onehot(goal, n_subgoals), (N, 1))
                    xs_goal = np.concatenate([flat_obs, goal_mat], axis=1)
                    log_components.append(np.log(high_probs[:-1, goal]) + self.regressor.predict_log_likelihood(
                        xs_goal, ys))
                log_components = np.array(log_components)
                log_p_sprime_given_s = np.log(np.sum(np.exp(log_components - log_components.max(axis=0)), axis=0)) + \
                    log_components.max(axis=0)
            elif isinstance(self.subgoal_space, Box):
                p_sprime_given_s = 0.
                unflat_obs = self.env_spec.observation_space.unflatten_n(flat_obs)
                # if subgoals are continuous, we'd need to sample instead of marginalize
                for _ in range(self.n_subgoal_samples):
                    goals, _ = self.policy.high_policy.get_actions(unflat_obs)
                    xs_goal = np.concatenate([flat_obs, self.subgoal_space.flatten_n(goals)], axis=1)
                    p_sprime_given_s += np.exp(self.regressor.predict_log_likelihood(xs_goal, ys))
                p_sprime_given_s /= self.n_subgoal_samples
                log_p_sprime_given_s = np.log(p_sprime_given_s + 1e-8)
            else:
                raise NotImplementedError
        bonuses = np.append(log_p_sprime_given_g_s - log_p_sprime_given_s, 0)
        bonuses = np.tile(
            np.expand_dims(bonuses, axis=1),
            (1, self.subgoal_interval)
        ).flatten()[:path_length]
        return bonuses
