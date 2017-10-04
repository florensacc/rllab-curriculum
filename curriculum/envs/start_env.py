"""
Start based environments. The classes inside this file should inherit the classes
from the state environment base classes.
"""


import random
from collections import OrderedDict

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

from curriculum.envs.base import StateGenerator, UniformListStateGenerator, \
    UniformStateGenerator, FixedStateGenerator, StateAuxiliaryEnv
from curriculum.experiments.asym_selfplay.algos.asym_selfplay import AsymSelfplay

from curriculum.experiments.asym_selfplay.algos.asym_selfplay_batch import AsymSelfplayBatch
from curriculum.experiments.asym_selfplay.envs.alice_env import AliceEnv
from curriculum.state.evaluator import parallel_map, FunctionWrapper
from curriculum.state.utils import StateCollection
from curriculum.logging.visualization import plot_labeled_states, plot_labeled_samples
from curriculum.state.evaluator import FunctionWrapper, parallel_map
from rllab.sampler.stateful_pool import singleton_pool


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

    def transform_to_start_space(self, obs, *args, **kwargs):
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

def generate_starts_random(starts=None, horizon = 1, size = 1000, subsample = None, sigma = 0.2,
                           center = None, range_lower = None, range_upper = None, noise = "gaussian"):
    # have to check that starts generated are feasible
    logger.log("generating random starts")
    if starts is None:
        raise Exception
    if noise == "gaussian":
        random_generator = random.gauss
    else:
        random_generator = random.uniform
    states = []
    if center is not None and range_lower is not None and range_upper is not None:
        center = np.array(center)
        range_lower = np.array(range_lower)
        range_upper = np.array(range_upper)
        while len(states) < size:
            start = starts[random.randint(0, len(starts) - 1)]
            new_state = np.random.randn(*start.shape) * sigma + start
            if np.all(new_state > center + range_lower) and np.all(new_state < center + range_upper):
                states.append(new_state)

    if subsample is None:
        return np.array(states)
    else:
        states = np.array(states)
        if len(states) < subsample:
            return states
        return states[np.random.choice(np.shape(states)[0], size=subsample)]

def generate_starts_alice(env_alice, algo_alice, log_dir, start_states=None, num_new_starts=10000,
                          start_generation=True, debug=False):

    asym_selfplay = AsymSelfplayBatch(algo_alice=algo_alice, env_alice=env_alice, start_states=start_states,
                                      num_rollouts=num_new_starts, log_dir=log_dir, start_generation=start_generation,
                                      debug=debug)

    # asym_selfplay = AsymSelfplay(algo_alice=algo_alice, algo_bob=None, env_alice=env_alice, env_bob=env_bob,
    #                              policy_alice=policy_alice, policy_bob=policy_bob, start_states=start_states,
    #                              num_rollouts=num_new_starts, alice_factor=alice_factor, alice_bonus=alice_bonus,
    #                              log_dir=log_dir)

    new_start_states, t_alices = asym_selfplay.optimize_batch()
    #new_start_states = asym_selfplay.optimize()
    logger.log('Done generating starts by Alice')
    return (np.array(new_start_states), t_alices)


def generate_starts(env, policy=None, starts=None, horizon=50, size=10000, subsample=None, variance=1,
                    zero_action=False, animated=False, speedup=1):
    """ If policy is None, brownian motion applied """
    if starts is None or len(starts) == 0:
        starts = [env.reset()]
    print("the starts from where we generate more is of len: ", len(starts))
    if horizon <= 1:
        states = starts  # you better give me some starts if there is no horizon!
    else:
        n_starts = len(starts)
        i = 0
        done = False
        obs = env.reset(init_state=starts[i % n_starts])
        states = [env.start_observation]
        steps = 0
        num_roll_reached_goal = 0
        num_roll = 0
        goal_reached = False
        # if animated:
        #     env.render()
        while len(states) < size:
            if animated:
                steps += 1
                if done or steps >= horizon:
                    i += 1
                    steps = 0
                    done = False
                    obs = env.reset(init_state=starts[i % n_starts])
                    # import pdb; pdb.set_trace()
                    states.append(env.start_observation)
                    num_roll += 1
                    if goal_reached:
                        num_roll_reached_goal += 1
                else:
                    noise = np.random.uniform(*env.action_space.bounds)
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
                # env.render()
                # timestep = 0.05
                # time.sleep(timestep / speedup)
            else:
                # import pdb; pdb.set_trace()
                brownian_state_wrapper = FunctionWrapper(
                    brownian,
                    env=env,
                    kill_outside=env.kill_outside,
                    kill_radius=env.kill_radius,  # this should be set before passing the env to generate_starts
                    horizon=horizon,
                    variance=variance,
                    policy=policy,
                )
                parallel_starts = [starts[j % n_starts] for j in range(i,i+singleton_pool.n_parallel)]
                # print("parallel sampling from :", parallel_starts)
                i += singleton_pool.n_parallel
                results = parallel_map(brownian_state_wrapper, parallel_starts)
                new_states = np.concatenate([result[0] for result in results])

                # show where these states are:
                np.random.shuffle(new_states)  # todo: this has a prety big impoact!! Why?? (related to collection)
                # generate_starts(env, starts=new_states, horizon=10, variance=0,
                #                 zero_action=True, animated=True, speedup=10, size=100)

                print('Just collected {} rollouts, with {} states'.format(len(results), new_states.shape))
                states.extend(new_states.tolist())
                print('now the states are of len: ', len(states))
                num_roll_reached_goal += np.sum([result[1] for result in results])
                print("num_roll_reached_goal ",  np.sum([result[1] for result in results]))
                num_roll += len(results)
        logger.log("Generating starts, rollouts that reached goal: " + str(num_roll_reached_goal) + " out of " + str(num_roll))
    logger.log("Starts generated.")
    if subsample is None:
        return np.stack([np.array(state) for state in states])
    else:
        states = np.stack([np.array(state) for state in states])
        if len(states) < subsample:
            return states
        return states[np.random.choice(np.shape(states)[0], size=subsample)]

def parallel_check_feasibility(starts, env, max_path_length=50, n_processes=-1):
    feasibility_wrapper = FunctionWrapper(
        check_feasibility,
        env =env,
        max_path_length=max_path_length,
    )
    is_feasible = parallel_map(
        feasibility_wrapper,
        starts,
        n_processes,
    )
    #TODO: is there better way to do this?
    result = [starts[i] for i in range(len(starts)) if is_feasible[i]] # keep starts that are feasible only
    return np.array(result)

def check_feasibility(start, env, max_path_length = 50):
    """
    Rolls out a policy with no action on ENV wifh init_state START for STEPS
    useful for checking if a state should be added to generated starts--if it's incredibly unstable, then a trained
    policy will likely not be able to work well
    :param env:
    :param steps:
    :return: True iff state is good
    """
    path_length = 0
    d = False
    o = env.reset(start)
    while path_length < max_path_length:
        a = np.zeros(env.action_space.flat_dim)
        next_o, r, d, env_info = env.step(a)
        path_length += 1
        if d:
            break
    return not d

def find_all_feasible_states_plotting(env, seed_starts, report, distance_threshold=0.1, size=10000, horizon = 300, brownian_variance=1, animate=False,
                                      num_samples = 100, limit = None, center = None, fast = True, check_feasible = True,
                                      check_feasible_path_length=50):
    """
    Generates states for two maze environments (ant and swimmer)
    :param env:
    :param seed_starts:
    :param report:
    :param distance_threshold: min distance between states
    :param size:
    :param horizon:
    :param brownian_variance:
    :param animate:
    :param num_samples: number of samples produced every iteration
    :param limit:
    :param center:
    :param fast:
    :param check_feasible:
    :param check_feasible_path_length:
    :return:
    """
    # If fast is True, we sample half the states from the last set generated and half from all previous generated
# label some states generated from last iteration and some from all
    log_dir = logger.get_snapshot_dir()
    if log_dir is None:
        log_dir = "/home/michael/"

    iteration = 0
    # use only first two coordinates (so in fransformed space
    all_feasible_starts = StateCollection(distance_threshold=distance_threshold, states_transform=lambda x: x[:, :2])
    all_feasible_starts.append(seed_starts)
    all_starts_samples = all_feasible_starts.sample(num_samples)
    text_labels =  OrderedDict({
        0: 'New starts',
        1: 'Old sampled starts',
        2: 'Other'
    })
    img = plot_labeled_samples(samples = all_starts_samples[:,:2],  # first two are COM
                        sample_classes=np.zeros(num_samples, dtype=int),
                               text_labels= text_labels,
                               limit=limit,
                               center=center,
                               maze_id=0,
                               )
    report.add_image(img, 'itr: {}\n'.format(iteration), width=500)
    report.save()

    no_new_states = 0
    while no_new_states < 30:
        iteration += 1
        logger.log("Iteration: {}".format(iteration))
        total_num_starts = all_feasible_starts.size
        starts = all_feasible_starts.sample(num_samples)

        # definitely want to initialize from new generated states, roughtly half proportion of both
        if fast and iteration > 1:
            print(len(added_states))
            if len(added_states) > 0:
                while len(starts) < 1.5 * num_samples:
                    starts = np.concatenate((starts, added_states), axis=0)
        new_starts = generate_starts(env, starts=starts, horizon=horizon, size=size, variance=brownian_variance,
                                     animated=animate, speedup=50)
        # filters starts so that we only keep the good starts
        if check_feasible: # used for ant maze environment, where we ant to run no_action
            logger.log("Prefilteredstarts: {}".format(len(new_starts)))
            new_starts = parallel_check_feasibility(env=env, starts=new_starts, max_path_length=check_feasible_path_length)
            # new_starts = [start for start in new_starts if check_feasibility(env, start, check_feasible_path_length)]
            logger.log("Filtered starts: {}".format(len(new_starts)))
        all_starts_samples = all_feasible_starts.sample(num_samples)
        added_states = all_feasible_starts.append(new_starts)
        num_new_starts = len(added_states)
        logger.log("number of new states: " + str(num_new_starts))
        if num_new_starts < 3:
            no_new_states += 1
        with open(osp.join(log_dir, 'all_feasible_states.pkl'), 'wb') as f:
            cloudpickle.dump(all_feasible_starts, f, protocol=3)


        # want to plot added_states and old sampled starts
        img = plot_labeled_samples(samples=np.concatenate((added_states[:, :2], all_starts_samples[:, :2]), axis = 0),  # first two are COM
                           sample_classes=np.concatenate((np.zeros(num_new_starts, dtype=int), np.ones(num_samples, dtype=int)), axis =0),
                                   text_labels=text_labels,
                                   limit=limit,
                                   center=center,
                                   maze_id=0,
                                   ) # fine if sample classes is longer


        report.add_image(img, 'itr: {}\n'.format(iteration), width=500)
        report.add_text("number of new states: " + str(num_new_starts))
        report.save()
        # break

    all_starts_samples = all_feasible_starts.sample(all_feasible_starts.size)
    img = plot_labeled_samples(samples=all_starts_samples,
                               # first two are COM
                               sample_classes=np.ones(all_feasible_starts.size, dtype=int),
                               text_labels=text_labels,
                               limit=limit,
                               center=center,
                               maze_id=0,
                               )  # fine if sample classes is longer

    report.add_image(img, 'itr: {}\n'.format(iteration), width=500)
    report.add_text("Total number of states: " + str(all_feasible_starts.size))
    report.save()


def brownian(start, env, kill_outside, kill_radius, horizon, variance, policy=None):
    # print('starting rollout from : ', start)
    with env.set_kill_outside(kill_outside=kill_outside, radius=kill_radius):
        done = False
        goal_reached = False
        steps = 0
        states = [start]
        obs = env.reset(start)
        while not done and steps < horizon:
            steps += 1
            noise = np.random.uniform(*env.action_space.bounds)
            if policy is not None:
                action, _ = policy.get_action(obs)
            else:
                action = noise
            obs, _, done, env_info = env.step(action)
            states.append(env.start_observation)
            if done and 'goal_reached' in env_info and env_info['goal_reached']:  # we don't care about goal done, otherwise will never advance!
                goal_reached = True
                done = False
    return states, goal_reached


def find_all_feasible_states(env, seed_starts, distance_threshold=0.1, brownian_variance=1, animate=False, speedup=10,
                             max_states = None, horizon = 1000, states_transform = None):
    # states_transform is optional transform of states
    # print('the seed_starts are of shape: ', seed_starts.shape)
    log_dir = logger.get_snapshot_dir()
    if states_transform is not None:
        all_feasible_starts = StateCollection(distance_threshold=distance_threshold, states_transform=states_transform)
    else:
        all_feasible_starts = StateCollection(distance_threshold=distance_threshold)
    all_feasible_starts.append(seed_starts)
    logger.log('finish appending all seed_starts')
    no_new_states = 0
    while no_new_states < 5:
        total_num_starts = all_feasible_starts.size
        if max_states is not None:
            if total_num_starts > max_states:
                return
        starts = all_feasible_starts.sample(100)
        new_starts = generate_starts(env, starts=starts, horizon=horizon, size=10000, variance=brownian_variance,
                                     animated=animate, speedup=speedup)
        logger.log("Done generating new starts")
        all_feasible_starts.append(new_starts, n_process=1)
        num_new_starts = all_feasible_starts.size - total_num_starts
        logger.log("number of new states: {}, total_states: {}".format(num_new_starts, all_feasible_starts.size))
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

