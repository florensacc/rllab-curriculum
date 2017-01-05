#!/usr/bin/env python

import argparse
import joblib
import os
import numpy as np
import random
import theano

from rllab.misc import ext
from rllab.sampler import parallel_sampler

N = 10  # Number of trajectory rollouts to perform
BATCH_SIZE = 50000  # Should be large enough to ensure that there are at least N trajs
DATA_DIR = "/home/shhuang/src/rllab-private/data/local/experiment/"
PARAMS_FNAME = 'params.pkl'

FGSM_EPS = 0.0001  # Amount to change each pixel
FGSM_RANDOM_SEED = 100001

def get_average_return(algo, n, seed=None):
    if seed is not None:
        # Set random seed, for reproducibility
        #np.random.seed(seed)
        #random.seed(seed)
        #ext.set_seed(seed)
        parallel_sampler.set_seed(seed)
        algo.env._wrapped_env.env._seed(seed)  # Set OpenAI AtariEnv seed

    paths = algo.sampler.obtain_samples(None)
    paths = paths[:n]
    assert len(paths) == n, "Not enough paths sampled -- increase BATCH_SIZE"
    avg_return = np.mean([sum(p['rewards']) for p in paths])
    return avg_return, paths

def fgsm_perturbation(obs, algo, fgsm_eps):
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
    return obs + fgsm_eps * sign_grad_x

def load_model(data_folder):
    # Load model from saved file
    data = joblib.load(os.path.join(DATA_DIR, data_folder, PARAMS_FNAME))
    algo = data['algo']
    algo.batch_size = BATCH_SIZE
    algo.max_path_length = data['env'].horizon

    # Copying what happens at the start of algo.train()
    algo.start_worker()
    algo.init_opt()
    return algo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder', type=str)
    args = parser.parse_args()

    # Load model from saved file
    algo = load_model(args.data_folder)

    # Run policy rollouts for N trajectories, get average return
    avg_return, paths = get_average_return(algo, N, seed=FGSM_RANDOM_SEED)
    print(avg_return)

    # Run policy rollouts with FGSM adversary for N trials, get average return
    if hasattr(algo.env, "_wrapped_env"):  # Means algo.env is a ProxyEnv
        algo.env._wrapped_env.set_adversary_fn(lambda x: fgsm_perturbation(x, algo, FGSM_EPS))
    else:
        algo.env.set_adversary_fn(lambda x: fgsm_perturbation(x, algo, FGSM_EPS))
    avg_return_adversary = get_average_return(algo, N, seed=FGSM_RANDOM_SEED)
    print(avg_return_adversary)

    # TODO: Visualize what the adversarial examples look like

if __name__ == "__main__":
    main()
