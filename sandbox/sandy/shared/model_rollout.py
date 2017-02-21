#!/usr/bin/env python

""" Get average return from rollouts of trained model
"""

import copy, numpy as np
from sandbox.sandy.envs.atari_env import get_base_env
<<<<<<< HEAD
=======
from sandbox.sandy.shared.ale_compatibility import set_gym_seed
>>>>>>> upstream/master

def set_seed_env(env, seed):
    from rllab.sampler import parallel_sampler
    #from rllab.misc import ext
    # Set random seed for policy rollouts
    #ext.set_seed(seed)
    parallel_sampler.set_seed(seed)

<<<<<<< HEAD
=======
    env.gym_seed = seed
>>>>>>> upstream/master
    # Set random seed of Atari environment
    if hasattr(env, 'ale'):  # envs/atari_env_haoran.py
        env.set_seed(seed)
    elif hasattr(env, '_wrapped_env'):  # Means env is a ProxyEnv
        base_env = get_base_env(env._wrapped_env)
<<<<<<< HEAD
        base_env._seed(seed)
    elif hasattr(env, 'env'):  # envs/atari_env.py
        base_env = get_base_env(env)
        base_env._seed(seed)
=======
        set_gym_seed(base_env, seed)
    elif hasattr(env, 'env'):  # envs/atari_env.py
        base_env = get_base_env(env)
        set_gym_seed(base_env, seed)
>>>>>>> upstream/master
    else:
        raise Exception("Invalid environment")

def set_seed(algo, seed):
    set_seed_env(algo.env, seed)

<<<<<<< HEAD
def get_average_return_trpo(algo, seed, N=10):
=======
def get_average_return_trpo(algo, seed, N=10, deterministic=False):
>>>>>>> upstream/master
    # Note that batch size is set during load_model

    # Set random seed, for reproducibility
    set_seed(algo, seed)
    curr_seed = seed + 1

    #paths = algo.sampler.obtain_samples(None)
    paths = []
    while len(paths) < N:
<<<<<<< HEAD
        new_paths = algo.sampler.obtain_samples(n_samples=1)  # Returns single path
=======
        new_paths = algo.sampler.obtain_samples(n_samples=1, deterministic=deterministic)  # Returns single path
>>>>>>> upstream/master
        paths.append(new_paths[0])
        set_seed(algo, curr_seed)
        curr_seed += 1

    avg_return = np.mean([sum(p['rewards']) for p in paths])
    timesteps = sum([len(p['rewards']) for p in paths])
    return avg_return, paths, timesteps

<<<<<<< HEAD
def get_average_return_a3c(algo, seed, N=10, horizon=10000):
=======
def get_average_return_a3c(algo, seed, N=10, horizon=10000, deterministic=False):
>>>>>>> upstream/master

    #scores, paths = algo.evaluate_performance(N, horizon, return_paths=True)
    paths = []
    curr_seed = seed
    while len(paths) < N:
        algo.test_env = copy.deepcopy(algo.cur_env)
        # copy.deepcopy doesn't copy lambda function
        algo.test_env.adversary_fn = algo.cur_env.adversary_fn
        algo.test_agent = copy.deepcopy(algo.cur_agent)
        # Set random seed, for reproducibility
        set_seed_env(algo.test_env, curr_seed)

<<<<<<< HEAD
        _, new_paths = algo.evaluate_performance(1, horizon, return_paths=True)
        paths.append(new_paths[0])

=======
        _, new_paths = algo.evaluate_performance(1, horizon, return_paths=True, \
                                                 deterministic=deterministic)
        paths.append(new_paths[0])

        # If algo.env is currently recording, make sure to update the states
        # recorded by algo.test_env
        #if algo.env.recording:
        #    algo.env.update_recording_from(algo.test_env)

>>>>>>> upstream/master
        del algo.test_env
        del algo.test_agent
        curr_seed += 1

    avg_return = np.mean([sum(p['rewards']) for p in paths])
    timesteps = sum([len(p['rewards']) for p in paths])
    return avg_return, paths, timesteps

def sample_dqn(algo, n_paths=1):  # Based on deep_q_rl/ale_experiment.py, run_episode
    env = algo.env
    #paths = [{'rewards':[], 'states':[], 'actions':[]} for i in range(n_paths)]
    rewards = [[] for i in range(n_paths)]
    timesteps = 0
    for i in range(n_paths):
        env.reset()
        action = algo.agent.start_episode(env.last_state)
        total_reward = 0
        while True:
            if env.is_terminal:
                algo.agent.end_episode(env.reward)
                break
            #paths[i]['states'].append(env.observation)
            #paths[i]['actions'].append(action)
            env.step(action)
            #paths[i]['rewards'].append(env.reward)
            total_reward += env.reward
            rewards[i].append(env.reward)
            action = algo.agent.step(env.reward, env.last_state, {})
            timesteps += 1
    return rewards, timesteps

<<<<<<< HEAD
def get_average_return_dqn(algo, seed, N=10):
=======
def get_average_return_dqn(algo, seed, N=10, deterministic=False):
    # deterministic - has no effect for DQN since (by definition) it selects
    # the argmax action

>>>>>>> upstream/master
    # Set random seed, for reproducibility
    set_seed(algo, seed)
    curr_seed = seed + 1

    paths = [{} for i in range(N)]
    total_timesteps = 0
    for i in range(N):
        rewards, timesteps = sample_dqn(algo, n_paths=1)  # Returns single path
        paths[i]['rewards'] = rewards[0]
        total_timesteps += timesteps
        set_seed(algo, curr_seed)
        curr_seed += 1

    avg_return = np.mean([sum(p['rewards']) for p in paths])
    return avg_return, paths, total_timesteps
    
<<<<<<< HEAD
def get_average_return(model, seed, N=10, return_timesteps=False):
    # model - instantiation of TrainedModel, or just the algo
=======
def get_average_return(model, seed, N=10, return_timesteps=False, deterministic=False):
    # model - instantiation of TrainedModel, or just the algo
    # deterministic - if True, then during test time model always picks argmax
    #         action to execute, rather than sampling output distribution over actions
>>>>>>> upstream/master
    if hasattr(model, 'algo'):
        algo = model.algo
    else:
        algo = model
    
    algo_name = type(algo).__name__
    if algo_name in ['TRPO', 'ParallelTRPO']:
        get_average_return_f = get_average_return_trpo
    elif algo_name in ['A3CALE']:
        get_average_return_f = get_average_return_a3c
    elif algo_name in ['DQNAlgo']:
        get_average_return_f = get_average_return_dqn
    else:
        raise NotImplementedError

<<<<<<< HEAD
    avg_return, paths, timesteps = get_average_return_f(algo, seed, N=N)
=======
    avg_return, paths, timesteps = get_average_return_f(algo, seed, N=N, deterministic=deterministic)
>>>>>>> upstream/master

    if return_timesteps:
        return avg_return, paths, timesteps
    else:
        return avg_return, paths
