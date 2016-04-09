from rllab.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.policies.subgoal_policy import SubgoalPolicy
from rllab.spaces.discrete import Discrete
from sandbox.rocky.grid_world_hrl_utils import ExactComputer
from rllab.misc import tensor_utils
from rllab import hrl_utils
from rllab.misc import logger
import numpy as np


class ExactStateGivenGoalMIEvaluator(object):
    def __init__(self, env, policy):
        self.exact_computer = ExactComputer(env, policy)
        # assert isinstance(env, CompoundActionSequenceEnv)
        # assert isinstance(env.wrapped_env, GridWorldEnv)
        # assert isinstance(policy, SubgoalPolicy)
        # assert isinstance(policy.subgoal_space, Discrete)
        # assert env._reset_history
        self.env = env
        self.policy = policy
        self.n_states = env.wrapped_env.observation_space.n
        self.n_raw_actions = env.wrapped_env.action_space.n
        self.n_subgoals = policy.subgoal_space.n
        self.subgoal_interval = policy.subgoal_interval
        self.computed = False

        self.p_next_state_given_goal_state = None
        self.p_next_state_given_state = None
        self.p_goal_given_state = None
        self.ent_next_state_given_goal_state = None
        self.ent_next_state_given_state = None
        self.mi_states = None
        self.mi_avg = None

    def _get_relevant_data(self, paths):
        obs = np.concatenate([p["agent_infos"]["high_obs"][:-1] for p in paths])
        next_obs = np.concatenate([p["agent_infos"]["high_obs"][1:] for p in paths])
        subgoals = np.concatenate([p["agent_infos"]["subgoal"][:-1] for p in paths])
        N = obs.shape[0]
        return obs.reshape((N, -1)), next_obs.reshape((N, -1)), subgoals

    def predict(self, path):
        path_length = len(path["rewards"])
        subsampled_path = hrl_utils.subsample_path(path, self.subgoal_interval)
        self.update_cache()
        flat_obs, flat_next_obs, subgoals = self._get_relevant_data([subsampled_path])
        obs = [self.env.observation_space.unflatten(o) for o in flat_obs]
        next_obs = [self.env.observation_space.unflatten(o) for o in flat_next_obs]
        subgoals = [self.policy.subgoal_space.unflatten(g) for g in subgoals]
        ret = np.log(self.p_next_state_given_goal_state[subgoals, obs, next_obs] + 1e-8) - np.log(
            self.p_next_state_given_state[obs, next_obs] + 1e-8)
        ret = np.append(ret, 0)
        ret = np.tile(
            np.expand_dims(ret, axis=1),
            (1, self.policy.subgoal_interval)
        ).flatten()[:path_length]
        return ret

    def update_cache(self):
        # We need to compute the quantity I(g, s'|s) = H(s'|s) - H(s'|g,s)
        # To compute this, we need to compute p(s'|g,s) and p(s'|s), which in turn needs p(g|s)
        # The second one is simply given by the policy. For the first one we need to do some work
        # We have p(s'|g,s) = sum_a p(s'|a,s) p(a|g,s)

        # We should only recompute when the policy changes

        if self.computed:
            return
        self.computed = True

        p_next_state_given_goal_state = self.exact_computer.compute_p_next_state_given_goal_state()
        p_goal_given_state = self.exact_computer.compute_p_goal_given_state()
        p_next_state_given_state = self.exact_computer.compute_p_next_state_given_state(
            p_next_state_given_goal_state=p_next_state_given_goal_state,
            p_goal_given_state=p_goal_given_state
        )

        # Now we can compute the entropies
        ent_next_state_given_state = self.exact_computer.compute_ent_next_state_given_state(p_next_state_given_state)
        ent_next_state_given_goal_state = self.exact_computer.compute_ent_next_state_given_goal_state(
            p_next_state_given_goal_state)
        mi_states = self.exact_computer.compute_mi_states(
            ent_next_state_given_goal_state=ent_next_state_given_goal_state,
            ent_next_state_given_state=ent_next_state_given_state,
            p_goal_given_state=p_goal_given_state
        )

        self.p_next_state_given_goal_state = p_next_state_given_goal_state
        self.p_next_state_given_state = p_next_state_given_state
        self.p_goal_given_state = p_goal_given_state
        self.ent_next_state_given_goal_state = ent_next_state_given_goal_state
        self.ent_next_state_given_state = ent_next_state_given_state
        self.mi_states = mi_states
        self.mi_avg = np.mean(mi_states)

    def log_diagnostics(self, paths):
        self.update_cache()
        logger.record_tabular("I(goal,next_state|state)", self.mi_avg)

    def fit(self, paths):
        # calling fit = invalidate caches
        self.computed = False
