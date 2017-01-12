#!/usr/bin/env python

import argparse
import h5py
import joblib
import os
import numpy as np
import random
import theano

from rllab.misc import ext
from rllab.sampler import parallel_sampler
from sandbox.sandy.misc.util import create_dir_if_needed, get_time_stamp

SAVE_OUTPUT = True  # Save adversarial perturbations to time-stamped h5 file
DEFAULT_OUTPUT_DIR = '/home/shhuang/src/rllab-private/data/local/rollouts'

N = 10  # Number of trajectory rollouts to perform
BATCH_SIZE = 50000  # Should be large enough to ensure that there are at least N trajs

#FGSM_EPS = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]  # Amount to change each pixel (1.0 / 256 = 0.00390625)
#FGSM_EPS = [0.004]  # Amount to change each pixel (1.0 / 256 = 0.00390625)
#FGSM_EPS = [0.008, 0.016, 0.032, 0.064, 0.128]
FGSM_EPS = [0.008]
FGSM_RANDOM_SEED = 0

OBS_MIN = -1  # minimum possible value for input x (NOTE: domain-specific)
OBS_MAX = +1  # maximum possible value for input x (NOTE: domain-specific)

def get_average_return(algo, n, seed=None):
    if seed is not None:  # Set random seed, for reproducibility
        # Set random seed for policy rollouts
        #ext.set_seed(seed)
        parallel_sampler.set_seed(seed)

        # Set random seed of Atari environment
        if hasattr(algo.env, 'ale'):  # envs/atari_env_haoran.py
            algo.env.set_seed(seed)
        elif hasattr(algo.env, '_wrapped_env'):  # Means algo.env is a ProxyEnv
            # Figure out which level contains ALE
            # (New version of Monitor in OpenAI gym adds an extra level of wrapping)
            if hasattr(algo.env._wrapped_env.env, 'ale'):
                algo.env._wrapped_env.env._seed(seed) 
            elif hasattr(algo.env._wrapped_env.env.env, 'ale'):
                algo.env._wrapped_env.env.env._seed(seed)
            elif hasattr(algo.env._wrapped_env.env.env.env, 'ale'):
                algo.env._wrapped_env.env.env.env._seed(seed)
            else:
                raise NotImplementedError
        elif hasattr(algo.env, 'env'):  # envs/atari_env.py
            if hasattr(algo.env.env, 'ale'):
                algo.env.env._seed(seed)
            elif hasattr(algo.env.env.env, 'ale'):
                algo.env.env.env._seed(seed)
            elif hasattr(algo.env.env.env.env, 'ale'):
                algo.env.env.env.env._seed(seed)
            else:
                raise NotImplementedError
        else:
            raise Exception("Invalid environment")

    paths = algo.sampler.obtain_samples(None)
    paths = paths[:n]
    assert len(paths) == n, "Not enough paths sampled -- increase BATCH_SIZE"
    avg_return = np.mean([sum(p['rewards']) for p in paths])
    return avg_return, paths

def fgsm_perturbation(obs, algo, fgsm_eps, obs_min, obs_max, output_h5=None):
    # Apply fast gradient sign method (FGSM):
    #     x + \epsilon * sign(\grad_x J(\theta, x, y))
    #     where \theta = policy params, x = obs, y = action

    # Calculate \grad_x J(\theta, x, y)
    flat_obs = algo.policy.observation_space.flatten(obs)[np.newaxis,:]
    grad_x = algo.optimizer._opt_fun["f_obs_grad"](flat_obs)[0,:]

    # Calculate sign(\grad_x J(\theta, x, y))
    sign_grad_x = np.sign(grad_x)

    # Unflatten
    sign_grad_x = algo.policy.observation_space.unflatten(sign_grad_x)

    # Can only adjust the last frame (not the earlier frames), so zero out
    # values in the earlier frames
    sign_grad_x[:-1,:] = 0
    adv_obs = obs + fgsm_eps * sign_grad_x

    # Clip pixels to be within range [obs_min, obs_max]
    adv_obs = np.minimum(obs_max, np.maximum(obs_min, adv_obs))
    if np.min(adv_obs) < -1 or np.max(adv_obs) > 1:
        print("Min", np.min(adv_obs), "Max", np.max(adv_obs))

    if output_h5 is not None:
        output_f = h5py.File(output_h5, 'r+')
        idx = len(output_f['rollouts'])
        g = output_f['rollouts'].create_group(str(idx))
        g['change_unscaled'] = sign_grad_x[-1,:]
        g['change'] = fgsm_eps * sign_grad_x[-1,:]
        g['orig_input'] = obs[-1,:]
        g['adv_input'] = adv_obs[-1,:]
        output_f.close()
    
    return adv_obs

def load_model(params_file):
    # Load model from saved file
    print("LOADING MODEL")
    data = joblib.load(params_file)
    algo = data['algo']
    algo.batch_size = BATCH_SIZE
    algo.sampler.worker_batch_size = BATCH_SIZE
    algo.n_parallel = 1
    try:
        algo.max_path_length = data['env'].horizon
    except NotImplementedError:
        algo.max_path_length = 50000

    # Copying what happens at the start of algo.train()
    assert type(algo).__name__ in ['TRPO', 'ParallelTRPO'], "Algo type not supported"
    if 'TRPO' == type(algo).__name__:
        algo.start_worker()

    algo.init_opt()

    if 'ParallelTRPO' in str(type(algo)):
        algo.init_par_objs()

    return algo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('params_file', type=str)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", type=str, default='fgsm-pong')
    args = parser.parse_args()

    # Load model from saved file
    algo = load_model(args.params_file)

    # Run policy rollouts for N trajectories, get average return
    #avg_return, paths = get_average_return(algo, N, seed=FGSM_RANDOM_SEED)
    #print("Return:", avg_return)

    for fgsm_eps in FGSM_EPS:
        if SAVE_OUTPUT:
            create_dir_if_needed(args.output_dir)
            output_h5 = os.path.join(args.output_dir, \
                                     args.prefix + '_' + get_time_stamp() + '.h5')
            print("Output h5 file:", output_h5)
            f = h5py.File(output_h5, 'w')
            f.create_group('rollouts')
            f['adv_type'] = 'fgsm'
            f.create_group('adv_params')
            f['adv_params']['eps'] = fgsm_eps
            f.close()

        # Run policy rollouts with FGSM adversary for N trials, get average return
        if hasattr(algo.env, "_wrapped_env"):  # Means algo.env is a ProxyEnv
            algo.env._wrapped_env.set_adversary_fn(lambda x: fgsm_perturbation(x, algo, fgsm_eps, OBS_MIN, OBS_MAX, output_h5=output_h5))
        else:
            algo.env.set_adversary_fn(lambda x: fgsm_perturbation(x, algo, fgsm_eps, OBS_MIN, OBS_MAX, output_h5=output_h5))
        avg_return_adversary, _ = get_average_return(algo, N, seed=FGSM_RANDOM_SEED)
        print("Adversary Return:", avg_return_adversary)
        print("\tAdversary Params:", fgsm_eps)

if __name__ == "__main__":
    main()
