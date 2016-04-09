import numpy as np

from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.distributions.categorical import Categorical
from rllab import hrl_utils
from rllab.misc import ext
from rllab.misc.special import to_onehot
from rllab.policies.subgoal_policy import SubgoalPolicy
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from rllab.regressors.product_regressor import ProductRegressor
from rllab.regressors.categorical_mlp_regressor import CategoricalMLPRegressor
from rllab.spaces.discrete import Discrete
from rllab.spaces.box import Box
from rllab.spaces.product import Product
import cPickle as pickle
from sandbox.rocky.grid_world_hrl_utils import ExactComputer
import itertools


class DenseStateGivenGoalMIEvaluator(LasagnePowered, Serializable):
    """
    Defines the reward bonus as the mutual information between the future state and the subgoal, given the current
    state: I(s',g|s) = H(s'|s) - H(s'|g,s) = E[log(p(s'|g,s)) - log(p(s'|s))]
    We train a neural network to fit a Gaussian to p(s'|g,s). Since we only have a discrete set of goals,
    we can marginalize over them to get p(s'|s):
    p(s'|s) = sum_g p(g|s)p(s'|g,s)


    This evaluator is "Dense" in that mutual information is computed per time step. Since the low-level policy
    actually receives a tick, the predictor predicts next state from current state, subgoal, together with the time
    tick, as opposed to just the current state and subgoal.
    """

    def __init__(
            self,
            env_spec,
            policy,
            # regressor_cls=None,
            regressor_args=None,
            logger_delegate=None):
        assert isinstance(policy, SubgoalPolicy)
        assert isinstance(policy.subgoal_space, Discrete)
        assert isinstance(policy.high_policy.distribution, Categorical)
        assert isinstance(policy.low_policy.distribution, Categorical)

        Serializable.quick_init(self, locals())
        # if regressor_cls is None:
        #     regressor_cls = GaussianMLPRegressor
        if regressor_args is None:
            regressor_args = dict()

        self.regressor_output_space = regressor_output_space = policy.high_env_spec.observation_space
        if isinstance(regressor_output_space, Discrete):
            self.regressor = CategoricalMLPRegressor(
                input_shape=(policy.low_env_spec.observation_space.flat_dim,),
                output_dim=regressor_output_space.flat_dim,
                name="(s'|g,s)",
                **regressor_args
            )
        elif isinstance(regressor_output_space, Box):
            raise NotImplementedError
        elif isinstance(regressor_output_space, Product):
            regressors = []
            for idx, subspace in enumerate(regressor_output_space.components):
                assert isinstance(subspace, Discrete)
                sub_regressor_args = pickle.loads(pickle.dumps(regressor_args))
                regressor = CategoricalMLPRegressor(
                    input_shape=(policy.low_env_spec.observation_space.flat_dim,),
                    output_dim=subspace.flat_dim,
                    name="(s'_%d|g,s)" % idx,
                    **sub_regressor_args
                )
                regressors.append(regressor)
            self.regressor = ProductRegressor(regressors)
        else:
            raise NotImplementedError

        self.subgoal_space = policy.subgoal_space
        self.subgoal_interval = policy.subgoal_interval
        self.logger_delegate = logger_delegate

    def _get_relevant_data(self, paths):
        high_obs = np.concatenate([p["agent_infos"]["high_obs"][:-1] for p in paths])
        subgoals = np.concatenate([p["agent_infos"]["subgoal"][:-1] for p in paths])
        counters = np.concatenate([p["agent_infos"]["counter"][:-1] for p in paths])
        low_obs = np.concatenate([p["agent_infos"]["low_obs"][:-1] for p in paths])
        next_high_obs = np.concatenate([p["agent_infos"]["high_obs"][1:] for p in paths])
        # N = len(low_obs)
        return dict(
            high_obs=high_obs,
            subgoals=subgoals,
            counters=counters,
            low_obs=low_obs,
            next_high_obs=next_high_obs,
        )

    # def _split_ys(self, ys):
    #     splitted_next_high_obs = [[] for _ in len(self.regressor_output_space.components)]
    #     for next_high_obs_entry in next_high_obs:
    #         for idx, x, component in zip(
    #             itertools.count(),
    #             self.regressor_output_space.unflatten(next_high_obs_entry),
    #             self.regressor_output_space.components
    #         ):
    #             splitted_next_high_obs[idx].append(component.flatten(x))
    #     return splitted_next_high_obs
    #
    # def _product_predict_log_likelihood(self, xs, ys):
    #     splitted_next_high_obs = self._split_next_high_obs(next_high_obs)
    #
    # def _product_fit(self, xs, ys):
    #     splitted_ys = self._split_next_high_obs(ys)
    #     for regressor, split_ys in zip(self.regressors, splitted_next_high_obs):
    #         regressor.fit(xs, split_ys)

    def fit(self, paths):
        low_obs, next_high_obs = ext.extract(
            self._get_relevant_data(paths),
            "low_obs", "next_high_obs"
        )
        # if isinstance(self.regressor_output_space, Product):
        #     self._product_fit(xs, ys)
        #
        # else:
        self.regressor.fit(low_obs, next_high_obs)
        if self.logger_delegate:
            self.logger_delegate.fit(paths)

    def log_diagnostics(self, paths):
        if self.logger_delegate:
            self.logger_delegate.log_diagnostics(paths)

    def predict(self, path):
        high_obs, counters, low_obs, next_high_obs = ext.extract(
            self._get_relevant_data([path]),
            "high_obs", "counters", "low_obs", "next_high_obs"
        )
        # flat_low_obs, flat_next_high_obs = self._get_relevant_data([path])
        N = len(low_obs)
        # xs = np.concatenate([flat_obs, subgoals], axis=1)
        # ys = flat_next_obs
        high_probs = path["agent_infos"]["high"]["prob"]
        log_p_sprime_given_g_s = self.regressor.predict_log_likelihood(low_obs, next_high_obs)
        p_sprime_given_s = 0.
        n_subgoals = self.subgoal_space.n
        for goal in range(n_subgoals):
            goal_mat = np.tile(to_onehot(goal, n_subgoals), (N, 1))
            xs_goal = np.concatenate([high_obs, goal_mat, counters], axis=1)
            p_sprime_given_s += high_probs[:-1, goal] * np.exp(self.regressor.predict_log_likelihood(xs_goal, next_high_obs))
        bonuses = np.append(log_p_sprime_given_g_s - np.log(p_sprime_given_s + 1e-8), 0)
        # bonuses = np.tile(
        #     np.expand_dims(bonuses, axis=1),
        #     (1, self.subgoal_interval)
        # ).flatten()[:path_length]
        return bonuses

    # def get_predicted_mi(self, env, policy):
    #     exact_computer = ExactComputer(env, policy)
    #     # We need to compute p(s'|g,s) approximately
    #
    #     n_subgoals = policy.subgoal_space.n
    #     n_states = env.observation_space.n
    #     # index: [0] -> goal, [1] -> state, [2] -> next state
    #     p_next_state_given_goal_state = np.zeros((n_subgoals, n_states, n_states))
    #
    #     for state in xrange(n_states):
    #         for subgoal in xrange(n_subgoals):
    #             for next_state in xrange(n_states):
    #                 xs = [
    #                     np.concatenate([env.observation_space.flatten(state), policy.subgoal_space.flatten(subgoal)])
    #                 ]
    #                 ys = [
    #                     env.observation_space.flatten(next_state)
    #                 ]
    #                 p_next_state_given_goal_state[subgoal, state, next_state] = np.exp(
    #                     self.regressor.predict_log_likelihood(xs, ys)[0])
    #
    #     p_goal_given_state = exact_computer.compute_p_goal_given_state()
    #     p_next_state_given_state = exact_computer.compute_p_next_state_given_state(
    #         p_next_state_given_goal_state=p_next_state_given_goal_state,
    #         p_goal_given_state=p_goal_given_state
    #     )
    #
    #     # Now we can compute the entropies
    #     ent_next_state_given_state = exact_computer.compute_ent_next_state_given_state(p_next_state_given_state)
    #     ent_next_state_given_goal_state = exact_computer.compute_ent_next_state_given_goal_state(
    #         p_next_state_given_goal_state)
    #     mi_states = exact_computer.compute_mi_states(
    #         ent_next_state_given_goal_state=ent_next_state_given_goal_state,
    #         ent_next_state_given_state=ent_next_state_given_state,
    #         p_goal_given_state=p_goal_given_state
    #     )
    #     return np.mean(mi_states)
