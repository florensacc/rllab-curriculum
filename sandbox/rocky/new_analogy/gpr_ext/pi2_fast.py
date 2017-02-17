import multiprocessing

import pyprind

from gpr.optimizer.common import *
from gpr.optimizer.lqr import *
import gpr.reward
import numpy as np
from scipy import interpolate

from sandbox.rocky.new_analogy.gpr_ext.fast_forward_dynamics import FastForwardDynamics

from rllab.misc import logger


def ceildiv(a, b):
    return -(-a // b)


def normpdf(x, mean, sd):
    var = float(sd) ** 2
    pi = 3.1415926
    denom = (2 * pi * var) ** .5
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    return num / denom


def expand_actions(u, T, skip):
    if skip == 1:
        return u
    if len(u.shape) == 3:
        shape = [T, u.shape[1], u.shape[2]]
        u = np.reshape(u, (u.shape[0], u.shape[1] * u.shape[2]))
    else:
        shape = [T, u.shape[1]]
    u_new = np.zeros((T, u.shape[1]))
    for i in range(u.shape[1]):
        spline_degree = min(u.shape[0] - 1, 3)
        tck = interpolate.splrep(range(0, T, skip), u[:, i], s=0, k=spline_degree)
        u_new[:, i] = interpolate.splev(range(T), tck, der=0)
    return np.reshape(u_new, shape)


def compute_update(P, T, skip, rewards_val, samples, params, step, dimu, k, K):
    S = np.zeros(P)

    new_k = np.zeros((ceildiv(T, skip), dimu))
    new_K = np.zeros((ceildiv(T, skip), dimu, dimu))
    temps = []
    KLs = []

    for t in reversed(range(ceildiv(T, skip))):
        S += rewards_val[t]

        for i in range(P):
            if math.isnan(S[i]):
                S[i] = -1E100
        if t == 0:
            best_particle = samples[:, S.argmax(), :]
        temp = 1E-100 if step == params['num_iterations'] - 1 else 1e-3
        while temp < 100:
            prob = softmax(S / temp)
            # refit Gaussains
            new_k[t] = np.sum(prob.reshape(-1, 1) * samples[t], 0)
            centered = samples[t] - new_k[t].reshape(1, -1)
            cov = np.matmul(centered.reshape(P, dimu, 1), centered.reshape(P, 1, dimu))
            assert (cov.shape == (P, dimu, dimu))
            new_K[t] = np.sum(prob.reshape(-1, 1, 1) * cov, 0)
            KL = kl_gaussians(k[t], K[t], new_k[t], new_K[t])
            if KL < params['max_kl'] or step == params['num_iterations'] - 1:
                break
            else:
                temp *= 2.
        temps.append(temp)
        KLs.append(KL)
    return S, new_k, new_K, best_particle, np.asarray(temps), KLs
    # k[...] = new_k
    # K[...] = new_K


def pi2(env, xinit, params):
    world = env.world
    T = env.horizon
    skip = params["skip"]
    P = params["particles"]
    dimu = world.dimu

    fast_forward_dynamics = params["extras"]["ffd"]

    def rollout(u):

        # progbar = pyprind.ProgBar(T)  # vec_env.num_envs)
        batch = u.shape[1]
        u_new = expand_actions(u, T, skip)
        x = {}
        s = {}
        x[0] = np.tile(xinit.reshape(1, -1), (batch, 1))
        losses = np.zeros((T, batch))
        fast_forward_dynamics.reset()
        for t in range(T):
            x[t + 1], losses[t], _ = fast_forward_dynamics(x[t], u_new[t], t=t)
        #     progbar.update()
        # if progbar.active:
        #     progbar.stop()
        losses_new = np.zeros((u.shape[0], u.shape[1]))
        for i in range(u.shape[0] - 1):
            losses_new[i, :] = np.sum(losses[(i * skip):((i + 1) * skip), :], 0)
        losses_new[u.shape[0] - 1, :] = np.sum(losses[((u.shape[0] - 1) * skip):, :], 0)
        return losses_new

    # PI2 (notation from https://arxiv.org/pdf/1610.00529v1.pdf)

    if params['correlated_noise']:
        k = np.zeros((ceildiv(T, skip), dimu))
        k[0] = world.get_stall_action(xinit)
    else:
        k = np.tile(world.get_stall_action(xinit), (ceildiv(T, skip), 1))

    best_particle = np.copy(k)
    K = params["init_cov"] * np.tile(np.eye(dimu).reshape(1, dimu, dimu), (ceildiv(T, skip), 1, 1))

    if params['pre_lqr'] > 0:
        k = expand_actions(k, T, skip)
        k = ilqr(env, xinit, replace(params, 'max_lqr_iterations', params['pre_lqr']), k)['u']
        k = k[::skip, :]

    def preprocess_actions(actions):
        if not params['correlated_noise']:
            return actions
        result = np.copy(actions)
        for t in range(1, ceildiv(T, skip)):
            result[t] += result[t - 1]
        return result

    if params["num_iterations"] == -1:
        k = np.random.randn(*k.shape)

    for step in range(params['num_iterations']):
        # sample
        samples = np.empty((ceildiv(T, skip), P, dimu))
        for t in range(ceildiv(T, skip)):
            samples[t] = np.random.multivariate_normal(k[t], K[t], (P,))
        samples[:, 0, :] = best_particle
        # compute reward
        actions = preprocess_actions(samples)
        rewards_val = rollout(actions)

        S, k[...], K[...], best_particle, temp, KLs = \
            compute_update(P=P, T=T, skip=skip, rewards_val=rewards_val, samples=samples, params=params, step=step,
                           dimu=dimu, k=k, K=K)
        # S = np.zeros(P)
        #
        # for t in reversed(range(ceildiv(T, skip))):
        #     S += rewards_val[t]
        #     for i in range(P):
        #         if math.isnan(S[i]):
        #             S[i] = -1E100
        #     if t == 0:
        #         best_particle = samples[:, S.argmax(), :]
        #     temp = 1E-100 if step == params['num_iterations'] - 1 else 1e-3
        #     while temp < 100:
        #         prob = softmax(S / temp)
        #         # refit Gaussains
        #         new_k[t] = np.sum(prob.reshape(-1, 1) * samples[t], 0)
        #         centered = samples[t] - new_k[t].reshape(1, -1)
        #         cov = np.matmul(centered.reshape(P, dimu, 1), centered.reshape(P, 1, dimu))
        #         assert (cov.shape == (P, dimu, dimu))
        #         new_K[t] = np.sum(prob.reshape(-1, 1, 1) * cov, 0)
        #         KL = kl_gaussians(k[t], K[t], new_k[t], new_K[t])
        #         if KL < params['max_kl'] or step == params['num_iterations'] - 1:
        #             break
        #         else:
        #             temp *= 2.
        #     k[...] = new_k
        #     K[...] = new_K
        # logger.record_tabular('Iteration', step)
        # logger.record_tabular('CovNorm', norm(K[:-1, :, :]))
        # logger.record_tabular_misc_stat('Return', S, placement='front')
        # logger.record_tabular_misc_stat('DiscountReturn', S_discount, placement='front')
        # logger.record_tabular_misc_stat('KL', KLs, placement='front')
        # logger.record_tabular_misc_stat('Temperature', temp, placement='front')
        # logger.record_tabular_misc_stat('FinalReward', rewards_val[-1], placement='front')
        # logger.dump_tabular()
        logger.log('PI2 iter: %d temp min: %f max: %f cov norm: %f reward best: %f worst: %f final reward '
                   'best: %f' % (
                       step, temp.min(), temp.max(), norm(K[:-1, :, :]), S.max(), S.min(), rewards_val[-1].max()))
    k = preprocess_actions(k)
    k = expand_actions(k, T, skip)
    if params['post_lqr'] > 0:
        k = ilqr(env, xinit, replace(params, 'max_lqr_iterations', params['post_lqr']), k)['u']

    return LinearController(env, xinit, k).sample_rollouts()
