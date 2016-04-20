from __future__ import print_function
from __future__ import absolute_import
import numpy as np

from sandbox.rocky.hrl import hrl_utils
from rllab.misc import logger
from sandbox.rocky.hrl.grid_world_hrl_utils import ExactComputer
from rllab.spaces.product import Product


class ContinuousExactStateBasedMIEvaluator(object):
    def __init__(self, env, policy, component_idx=None):
        # self.exact_computer = ExactComputer(env, policy, component_idx)
        self.env = env
        self.policy = policy
        self.subgoal_space = policy.subgoal_space
        self.subgoal_interval = policy.subgoal_interval
        self.component_idx = component_idx
        if component_idx is None:
            self.component_space = self.env.observation_space
        else:
            assert isinstance(self.env.observation_space, Product)
            self.component_space = self.env.observation_space.components[component_idx]
        self.computed = False

        self.p_next_state_given_goal_state = None
        self.p_next_state_given_state = None
        self.p_goal_given_state = None
        self.ent_next_state_given_goal_state = None
        self.ent_next_state_given_state = None
        self.mi_states = None
        self.mi_avg = None

    def get_p_sprime_given_s_g(self, state, goal, next_state):
        int_state = self.env.analyzer.get_int_state_from_obs(state)
        int_next_component_state = self.env.analyzer.get_int_component_state_from_obs(next_state, self.component_idx)

        # for next_state in xrange(self.env.analyzer.n_states):

        self.env.analyzer.get_posterior_sequences(self.subgoal_interval, int_state, next_state)
        import ipdb; ipdb.set_trace()
        return 0

    def get_p_sprime_given_s(self, state, next_state):
        import ipdb; ipdb.set_trace()
        return 0

    def _get_relevant_data(self, paths):
        obs = np.concatenate([p["agent_infos"]["high_obs"][:-1] for p in paths])
        N = obs.shape[0]
        next_obs = np.concatenate([p["agent_infos"]["high_obs"][1:] for p in paths]).reshape((N, -1))
        if self.component_idx is not None:
            obs_flat_dims = [c.flat_dim for c in self.env.observation_space.components]
            slice_start = sum(obs_flat_dims[:self.component_idx])
            slice_end = slice_start + obs_flat_dims[self.component_idx]
            next_component_obs = next_obs[:, slice_start:slice_end]
        else:
            next_component_obs = next_obs
        subgoals = np.concatenate([p["agent_infos"]["subgoal"][:-1] for p in paths])
        return obs.reshape((N, -1)), next_component_obs, subgoals

    def predict(self, path):
        path_length = len(path["rewards"])
        subsampled_path = hrl_utils.downsample_path(path, self.subgoal_interval)
        self.update_cache()
        flat_obs, flat_next_obs, flat_subgoals = self._get_relevant_data([subsampled_path])
        obs = self.env.observation_space.unflatten_n(flat_obs)
        # int_obs = map(self.env.analyzer.get_int_state_from_obs, obs)
        next_obs = self.component_space.unflatten_n(flat_next_obs)
        # int_next_obs = [self.env.analyzer.get_int_component_state_from_obs(x, self.component_idx) for x in
        #             next_obs]
        subgoals = self.policy.subgoal_space.unflatten_n(flat_subgoals)
        ret = []
        for o, no, g in zip(obs, next_obs, subgoals):
            ret.append(np.log(self.get_p_sprime_given_s_g(o, g, no) + 1e-8) -
                       np.log(self.get_p_sprime_given_s(o, no) + 1e-8))
        ret.append(0)


        # ret = np.log(self.p_next_state_given_goal_state[subgoals, obs, next_obs] + 1e-8) - np.log(
        #     self.p_next_state_given_state[obs, next_obs] + 1e-8)
        # ret = np.append(ret, 0)
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

        # p_next_state_given_goal_state = self.exact_computer.compute_p_next_state_given_goal_state()
        # p_goal_given_state = self.exact_computer.compute_p_goal_given_state()
        # p_next_state_given_state = self.exact_computer.compute_p_next_state_given_state(
        #     p_next_state_given_goal_state=p_next_state_given_goal_state,
        #     p_goal_given_state=p_goal_given_state
        # )
        #
        # # Now we can compute the entropies
        # ent_next_state_given_state = self.exact_computer.compute_ent_next_state_given_state(p_next_state_given_state)
        # ent_next_state_given_goal_state = self.exact_computer.compute_ent_next_state_given_goal_state(
        #     p_next_state_given_goal_state)
        # mi_states = self.exact_computer.compute_mi_states(
        #     ent_next_state_given_goal_state=ent_next_state_given_goal_state,
        #     ent_next_state_given_state=ent_next_state_given_state,
        #     p_goal_given_state=p_goal_given_state
        # )
        #
        # self.p_next_state_given_goal_state = p_next_state_given_goal_state
        # self.p_next_state_given_state = p_next_state_given_state
        # self.p_goal_given_state = p_goal_given_state
        # self.ent_next_state_given_goal_state = ent_next_state_given_goal_state
        # self.ent_next_state_given_state = ent_next_state_given_state
        # self.mi_states = mi_states
        # self.mi_avg = np.mean(mi_states)

    # def mi_bonus_sym(self):
    #     return self.exact_computer.mi_bonus_sym()

    def log_diagnostics(self, paths):
        self.update_cache()
        # logger.record_tabular("I(goal,next_state|state)", self.mi_avg)

    def fit(self, paths):
        # calling fit = invalidate caches
        self.computed = False
        # self.exact_computer.prepare_sym(paths)
