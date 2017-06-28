"""
Start based environments. The classes inside this file should inherit the classes
from the state environment base classes.
"""


import random
from rllab import spaces
import sys
import os.path as osp
import cloudpickle
import pickle

import numpy as np
import scipy.misc
import tempfile
import math
import time

from rllab.algos.trpo import TRPO
from rllab.envs.mujoco.mujoco_env import MODEL_DIR, BIG
from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
from rllab.envs.base import Step
from rllab.misc import autoargs
from rllab.misc import logger
from rllab.sampler.utils import rollout
from rllab.spaces.box import Box
from rllab.misc.overrides import overrides

from sandbox.young_clgan.envs.base import StateGenerator, UniformListStateGenerator, \
    UniformStateGenerator, FixedStateGenerator, StateAuxiliaryEnv
from sandbox.young_clgan.experiments.asym_selfplay.algos.asym_selfplay import AsymSelfplay
from sandbox.young_clgan.experiments.asym_selfplay.envs.stop_action_env import AliceEnv
from sandbox.young_clgan.state.utils import StateCollection


class StartEnv(Serializable):
    """ A wrapper of StateAuxiliaryEnv to make it compatible with the old goal env."""

    def __init__(self, start_generator=None, append_start=False, obs2start_transform=None, *args, **kwargs):
        Serializable.quick_init(self, locals())
        self._start_holder = StateAuxiliaryEnv(state_generator=start_generator, *args, **kwargs)
        self.append_start = append_start
        if obs2start_transform is None:
            self._obs2start_transform = lambda x: x
        else:
            self._obs2start_transform = obs2start_transform

    def transform_to_start_space(self, obs):
        """ Apply the start space transformation to the given observation. """
        return self._obs2start_transform(obs)

    def update_start_generator(self, *args, **kwargs):
        # print("updating start generator with ", *args, **kwargs)
        return self._start_holder.update_state_generator(*args, **kwargs)
        
    def update_start(self, start=None, *args, **kwargs):
        return self._start_holder.update_aux_state(state=start, *args, **kwargs)
        
    @property
    def start_generator(self):
        return self._start_holder.state_generator
    
    @property
    def current_start(self):
        return self._start_holder.current_aux_state

    @property
    def start_observation(self):
        """ Get the start space part of the current observation. """
        obj = self
        while hasattr(obj, "wrapped_env"):  # try to go through "Normalize and Proxy and whatever wrapper"
            obj = obj.wrapped_env
        return self.transform_to_start_space(obj.get_current_obs())

    def append_start_observation(self, obs):
        """ Append current start to the given original observation """
        if self.append_start:
            return np.concatenate([obs, np.array(self.current_start)])
        else:
            return obs

    def __getstate__(self):
        d = super(StartEnv, self).__getstate__()
        d['__start_holder'] = self._start_holder
        return d

    def __setstate__(self, d):
        super(StartEnv, self).__setstate__(d)
        self._start_holder = d['__start_holder']


class StartExplorationEnv(StartEnv, ProxyEnv, Serializable):
    def __init__(self, env, start_generator, only_feasible=False, start_bounds=None, *args, **kwargs):
        """
        This environment wraps around a normal environment to facilitate goal based exploration.
        Initial position based experiments should not use this class.
        
        :param env: wrapped env
        :param start_generator: a StateGenerator object
        :param obs_transform: a callable that transforms an observation of the wrapped environment into goal space
        :param terminal_eps: a threshold of distance that determines if a goal is reached
        :param terminate_env: a boolean that controls if the environment is terminated with the goal is reached
        :param start_bounds: array marking the UB of the rectangular limit of goals.
        :param distance_metric: L1 or L2 or a callable func
        :param goal_weight: coef of the goal based reward
        :param inner_weight: coef of the inner environment reward
        :param append_transformed_obs: append the transformation of the current observation to full observation
        """
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        StartEnv.__init__(self, *args, **kwargs)
        self.update_start_generator(start_generator)
        
        self.start_bounds = start_bounds
        self.only_feasible = only_feasible
        
        # # TODO fix this
        # if self.start_bounds is None:
        #     self.start_bounds = self.wrapped_env.observation_space.bounds[1]  # we keep only UB
        #     self._feasible_start_space = self.wrapped_env.observation_space
        # else:
        #     self._feasible_start_space = Box(low=-1 * self.start_bounds, high=self.start_bounds)

    # @property
    # @overrides
    # def feasible_start_space(self):
    #     return self._feasible_start_space
    #
    # def is_feasible(self, start):
    #     obj = self.wrapped_env
    #     while not hasattr(obj, 'is_feasible') and hasattr(obj, 'wrapped_env'):
    #         obj = obj.wrapped_env
    #     if hasattr(obj, 'is_feasible'):
    #         return obj.is_feasible(np.array(start))  # but the goal might not leave in the same space!
    #     else:
    #         return True

    def reset(self, *args, **kwargs):
        self.update_start(*args, **kwargs)
        return self.wrapped_env.reset(init_state=self.current_start)


def generate_starts_alice(env_bob, env_alice, policy_bob, policy_alice, algo_alice, start_states=None,
                          num_new_starts=10000, alice_factor=0.5):
    asym_selfplay = AsymSelfplay(algo_alice=algo_alice, algo_bob=None, env_alice=env_alice, env_bob=env_bob,
                                 policy_alice=policy_alice, policy_bob=policy_bob, start_states=start_states,
                                 num_rollouts=num_new_starts, alice_factor=alice_factor)

    new_start_states = asym_selfplay.optimize_batch()
    logger.log('Done generating starts by Alice')
    return new_start_states

def generate_starts(env, policy=None, starts=None, horizon=50, size=10000, subsample=None, variance=1,
                    zero_action=False, animated=False, speedup=1):
    """ If policy is None, brownian motion applied """
    if starts is None or len(starts) == 0:
        starts = [env.reset()]
    if horizon <= 1:
        states = starts  # you better give me some starts if there is no horizon!
    else:
        n_starts = len(starts)
        i = 0
        done = False
        obs = env.reset(init_state=starts[i % n_starts])
        states = [env.start_observation]
        steps = 0
        noise = 0
        num_roll_reached_goal = 0
        num_roll = 0
        goal_reached = False
        if animated:
            env.render()
        while len(states) < size:
            steps += 1
            # print(steps)
            if done or steps >= horizon:
                steps = 0
                noise = 0
                i += 1
                done = False
                obs = env.reset(init_state=starts[i % n_starts])
                # print(obs)
                states.append(env.start_observation)
                num_roll += 1
                if goal_reached:
                    num_roll_reached_goal += 1
            else:
                noise += np.random.randn(env.action_space.flat_dim) * variance
                if policy:
                    action, _ = policy.get_action(obs)
                else:
                    action = noise
                if zero_action:
                    action = np.zeros_like(action)
                obs, _, done, env_info = env.step(action)
                states.append(env.start_observation)
                if done and env_info['goal_reached']:  # we don't care about goal done, otherwise will never advance!
                    goal_reached = True
                    done = False
            if animated:
                env.render()
                timestep = 0.05
                time.sleep(timestep / speedup)

        logger.log("Generating starts, rollouts that reached goal: " + str(num_roll_reached_goal) + " out of " + str(num_roll))
    if subsample is None:
        return np.array(states)
    else:
        states = np.array(states)
        if len(states) < subsample:
            return states
        return states[np.random.choice(np.shape(states)[0], size=subsample)]


def find_all_feasible_states(env, seed_starts, distance_threshold=0.1, brownian_variance=1, animate=False):
    log_dir = logger.get_snapshot_dir()
    all_feasible_starts = StateCollection(distance_threshold=distance_threshold)
    all_feasible_starts.append(seed_starts)
    no_new_states = 0
    while no_new_states < 5:
        total_num_starts = all_feasible_starts.size
        starts = all_feasible_starts.sample(100)
        new_starts = generate_starts(env, starts=starts, horizon=1000, size=100000, variance=brownian_variance,
                                     animated=animate, speedup=10)
        all_feasible_starts.append(new_starts)
        num_new_starts = all_feasible_starts.size - total_num_starts
        logger.log("number of new states: " + str(num_new_starts))
        if num_new_starts < 10:
            no_new_states += 1
        with open(osp.join(log_dir, 'all_feasible_states.pkl'), 'wb') as f:
            cloudpickle.dump(all_feasible_starts, f, protocol=3)


def find_all_feasible_reject_states(env, distance_threshold=0.1,):
    # test reject see how many are feasible
    uniform_state_generator = UniformStateGenerator(state_size=len(env.current_start), bounds=env.start_generator.bounds)
    any_starts = StateCollection(distance_threshold=distance_threshold)
    k = 0
    while any_starts.size < 1e6:
        state = uniform_state_generator.update()
        obs = env.reset(init_state=state)
        action = np.zeros(env.action_dim)
        next_obs, _, done, env_info = env.step(action)
        if not np.linalg.norm(next_obs - obs) == 0:
            print("CONTACT! obs changed:", obs, next_obs)
        elif done and not env_info['gaol_reached']:
            print("outside range")
        else:
            any_starts.append(state)
            print("any_starts: ", any_starts.size, " out of ", k)
        k += 1

#
#
# def evaluate_start_env(env, policy, horizon, n_starts=10, n_traj=1, **kwargs):
#     paths = [rollout(env=env, agent=policy, max_path_length=horizon) for _ in range(int(n_starts))]
#     env.log_diagnostics(paths, n_traj=n_traj, **kwargs)

