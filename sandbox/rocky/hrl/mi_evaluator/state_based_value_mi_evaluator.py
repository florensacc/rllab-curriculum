from __future__ import absolute_import
from __future__ import print_function

import numba
import numpy as np

from rllab.core.lasagne_powered import LasagnePowered
from rllab.core.serializable import Serializable
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor
from sandbox.rocky.hrl.policies.subgoal_policy import SubgoalPolicy


@numba.jit
def downsampled_discount_cumsum(x, discount, subgoal_interval):
    result = np.empty_like(x)
    cur_sum = 0
    terminal = len(x) - 1
    for t in xrange(terminal, -1, -1):
        multiplier = discount if (t % subgoal_interval == subgoal_interval - 1) else 1.
        cur_sum = cur_sum * multiplier + x[t]
        result[t] = cur_sum
    return result


class StateBasedValueMIEvaluator(LasagnePowered, Serializable):
    """
    Defines the reward bonus as the mutual information between the future cumulative reward and the subgoal,
    conditioned on the current state: I(sum(r), g|s) = H(sum(r)|s) - H(sum(r)|s,g). This boils down to training
    """

    def __init__(
            self,
            env,
            policy,
            discount,
            state_regressor_cls=None,
            state_regressor_args=None,
            state_goal_regressor_cls=None,
            state_goal_regressor_args=None):
        assert isinstance(policy, SubgoalPolicy)
        Serializable.quick_init(self, locals())
        self.subgoal_interval = policy.subgoal_interval
        self.discount = discount

        if state_regressor_args is None:
            state_regressor_args = dict()
        if state_regressor_cls is None:
            state_regressor_cls = GaussianMLPRegressor
        if state_goal_regressor_args is None:
            state_goal_regressor_args = dict()
        if state_goal_regressor_cls is None:
            state_goal_regressor_cls = GaussianMLPRegressor

        state_regressor = state_regressor_cls(
            input_shape=(env.observation_space.flat_dim,),
            output_dim=1,
            name="V|s",
            **state_regressor_args
        )

        state_goal_regressor = state_goal_regressor_cls(
            input_shape=(env.observation_space.flat_dim + policy.subgoal_space.flat_dim,),
            output_dim=1,
            name="V|s,g",
            **state_goal_regressor_args
        )

        self.state_regressor = state_regressor
        self.state_goal_regressor = state_goal_regressor

    def _extract_xys(self, paths):
        discount = self.discount
        subgoal_interval = self.subgoal_interval
        observations = np.concatenate([p["agent_infos"]["high_obs"] for p in paths])
        discount_returns = np.concatenate([downsampled_discount_cumsum(p["rewards"], discount, subgoal_interval) for
                                           p in paths])
        subgoals = np.concatenate([p["agent_infos"]["subgoal"] for p in paths])
        ys = np.reshape(discount_returns, (-1, 1))

        xs_states = observations
        xs_state_goals = np.concatenate([observations, subgoals], axis=1)
        return xs_states, xs_state_goals, ys

    def fit(self, paths):
        xs_states, xs_state_goals, ys = self._extract_xys(paths)

        self.state_regressor.fit(xs_states, ys)
        self.state_goal_regressor.fit(xs_state_goals, ys)

    def predict(self, path):
        xs_states, xs_state_goals, ys = self._extract_xys([path])
        log_p_v_given_s = self.state_regressor.predict_log_likelihood(xs_states, ys)
        log_p_v_given_s_g = self.state_goal_regressor.predict_log_likelihood(xs_state_goals, ys)
        ret = log_p_v_given_s_g - log_p_v_given_s
        return ret

    def log_diagnostics(self, paths):
        pass
