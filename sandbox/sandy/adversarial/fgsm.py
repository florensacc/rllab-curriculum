#!/usr/bin/env python

import argparse
import chainer
from chainer import functions as F
import numpy as np

from sandbox.sandy.adversarial.io_util import init_output_file, save_rollout_step
from sandbox.sandy.adversarial.shared import get_average_return, load_model

SUPPORTED_NORMS = ['l1', 'l2', 'l-inf']
SAVE_OUTPUT = True  # Save adversarial perturbations to time-stamped h5 file
DEFAULT_OUTPUT_DIR = '/home/shhuang/src/rllab-private/data/local/rollouts'

N = 10  # Number of trajectory rollouts to perform
BATCH_SIZE = 20000  # Should be large enough to ensure that there are at least N trajs

FGSM_EPS = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]  # Amount to change each pixel (1.0 / 256 = 0.00390625)
#FGSM_EPS = [0.004]  # Amount to change each pixel (1.0 / 256 = 0.00390625)
#FGSM_EPS = [0.008, 0.016, 0.032, 0.064, 0.128]
#FGSM_EPS = [0.004, 0.008, 0.016, 0.032]
FGSM_RANDOM_SEED = 0

OBS_MIN = -1  # minimum possible value for input x (NOTE: domain-specific)
OBS_MAX = +1  # maximum possible value for input x (NOTE: domain-specific)

def fgsm_perturbation_linf(grad_x, fgsm_eps):
    sign_grad_x = np.sign(grad_x)
    return fgsm_eps * sign_grad_x, sign_grad_x

def fgsm_perturbation_l2(grad_x, fgsm_eps):
    grad_x_unit = grad_x / np.linalg.norm(grad_x)
    scaled_fgsm_eps = fgsm_eps * np.sqrt(grad_x.size)
    return scaled_fgsm_eps * grad_x_unit, grad_x_unit

def fgsm_perturbation_l1(grad_x, fgsm_eps, obs, obs_min, obs_max):
    grad_x_flat = grad_x.flatten(order='C')
    obs_flat = obs.flatten(order='C')

    sorted_idxs = np.argsort(np.abs(grad_x_flat,))[::-1]
    budget = fgsm_eps * grad_x.size
    eta = np.zeros(sorted_idxs.shape)
    for idx in sorted_idxs:
        if budget <= 0:
            break
        diff = 0
        if grad_x_flat[idx] < 0:
            eta[idx] = obs_min - obs_flat[idx]
        elif grad_x_flat[idx] > 0:
            eta[idx] = obs_max - obs_flat[idx]
        if abs(eta[idx]) > budget:
            eta[idx] = np.sign(eta[idx]) * budget
        budget -= abs(eta[idx])

    if budget > 0:
        print("WARNING: L1 budget not completely used - epsilon larger than necessary")
    eta = eta.reshape(grad_x.shape, order='C')
    return eta, np.sign(eta)

def get_grad_x_a3c(obs, algo):
    statevar = chainer.Variable(np.expand_dims(algo.cur_agent.preprocess(obs), 0))
    logits = algo.cur_agent.model.pi.compute_logits(algo.cur_agent.model.head(statevar))
    max_logits = F.broadcast_to(F.max(logits), (1,len(logits)))
    # Calculate loss between predicted action distribution and the action distribution
    # that places all weight on the argmax action
    ce_loss = -1 * F.log(1.0 / F.sum(F.exp(logits - max_logits)))
    ce_loss.backward(retain_grad=True)
    grad_x = statevar.grad[0]
    return grad_x

def get_grad_x_trpo(obs, algo):
    flat_obs = algo.policy.observation_space.flatten(obs)[np.newaxis,:]
    grad_x = algo.optimizer._opt_fun["f_obs_grad"](flat_obs)[0,:]
    grad_x = algo.policy.observation_space.unflatten(grad_x)
    return grad_x

def fgsm_perturbation(obs, algo, **kwargs):
    # Apply fast gradient sign method (FGSM):
    # For l-inf norm,
    #     \eta =  \epsilon * sign(\grad_x J(\theta, x, y))
    #         where \theta = policy params, x = obs, y = action
    # For l2 norm,
    #     \eta = (\epsilon / ||\grad_x J(\theta, x, y)||_2) * \grad_x J(\theta, x, y)
    # For l1 norm,
    #     going down list of indices i ranked by |\grad_x J(\theta, x, y)|_i,
    #     maximally perturb \eta_i (to obs_min or obs_max, depending on sign
    #     of (\grad_x J(\theta, x, y))_i; have 'budget' of \epsilon total perturbation

    try:
        fgsm_eps = kwargs['fgsm_eps']
        norm = kwargs['norm']
        obs_min = kwargs['obs_min']
        obs_max = kwargs['obs_max']
        output_h5 = kwargs.get('output_h5', None)
    except KeyError:
        print("FGSM requires the following inputs: fgsm_eps, norm, obs_min, obs_max")
        raise

    # Calculate \grad_x J(\theta, x, y)
    algo_name = type(algo).__name__
    if algo_name in ['TRPO', 'ParallelTRPO']:
        grad_x = get_grad_x_trpo(obs, algo)
    elif algo_name in ['A3CALE']:
        grad_x = get_grad_x_a3c(obs, algo)
    else:
        assert False, "Algorithm type " + algo_name + " is not supported."
    grad_x_current = grad_x[-1,:,:]

    if norm == 'l-inf':
        eta, unscaled_eta = fgsm_perturbation_linf(grad_x_current, fgsm_eps)
    elif norm == 'l2':
        eta, unscaled_eta = fgsm_perturbation_l2(grad_x_current, fgsm_eps)
    elif norm == 'l1':
        eta, unscaled_eta = fgsm_perturbation_l1(grad_x_current, fgsm_eps, \
                                                 obs[-1,:,:], obs_min, obs_max)
    else:
        raise NotImplementedError

    # The computed perturbation is stored in eta: x_adversarial = x + eta
    # Only current frame can be changed by adversary (i.e., the last frame)
    adv_obs = np.array(obs)
    adv_obs[-1,:,:] += eta
    # Clip pixels to be within range [obs_min, obs_max]
    adv_obs = np.minimum(obs_max, np.maximum(obs_min, adv_obs))

    if output_h5 is not None:
        save_rollout_step(output_h5, eta, unscaled_eta, obs[-1,:,:], adv_obs[-1,:,:])
    return adv_obs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('params_file', type=str)
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", type=str, default='fgsm-pong')
    parser.add_argument("--norm", type=str, default='l-inf')
    args = parser.parse_args()

    assert args.norm in SUPPORTED_NORMS, "Norm must be one of: " + str(SUPPORTED_NORMS)

    # Load model from saved file
    algo, env = load_model(args.params_file, BATCH_SIZE)

    # Run policy rollouts for N trajectories, get average return
    #avg_return, paths = get_average_return(algo, FGSM_RANDOM_SEED, N)
    #print("Return:", avg_return)

    for fgsm_eps in FGSM_EPS:
        if SAVE_OUTPUT:
            output_h5 = init_output_file(args.output_dir, args.prefix, 'fgsm', \
                                         {'eps': fgsm_eps, 'norm': args.norm})
            print("Output h5 file:", output_h5)

        # Run policy rollouts with FGSM adversary for N trials, get average return
        env.set_adversary_fn(lambda x: fgsm_perturbation(x, algo, \
                                       fgsm_eps=fgsm_eps, obs_min=OBS_MIN, \
                                       obs_max=OBS_MAX, \
                                       output_h5=output_h5, norm=args.norm))
        avg_return_adversary, _ = get_average_return(algo, FGSM_RANDOM_SEED, N)
        print("Adversary Return:", avg_return_adversary)
        print("\tAdversary Params:", fgsm_eps, args.norm)

if __name__ == "__main__":
    main()
