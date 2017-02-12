import copy
import math
import multiprocessing

import numpy as np
from scipy import interpolate

from gpr.optimizer.common import LinearController, softmax, kl_gaussians
from rllab.misc import logger
from sandbox.rocky.new_analogy.gpr_ext.fast_forward_dynamics import FastForwardDynamics
import gpr.reward


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


class PI2(object):
    def __init__(
            self,
            env,
            xinit=None,
            init_k=None,
            time_multiplier=None,
            num_iterations=10,
            skip=1,
            particles=100,
            correlated_noise=False,
            init_cov=1.,
            # ctrl_smoothness_pen=0.,
            n_parallel=multiprocessing.cpu_count(),
            max_kl=100):
        """

        :param env:
        :param skip: "skip" points during sampling for pi2. Interpolates between them with splines.
        :param particles: number of trajectories sampled per iteration
        :param correlated_noise: use correlated samples
        :return:
        """
        self.env = env
        self.num_iterations = num_iterations
        self.skip = skip
        self.particles = particles
        self.correlated_noise = correlated_noise
        self.max_kl = max_kl
        self.init_cov = init_cov
        self.time_multiplier = time_multiplier
        # self.ctrl_smoothness_pen = ctrl_smoothness_pen
        self.xinit = xinit
        self.init_k = init_k
        self.n_parallel = n_parallel

    def train(self):
        logger.log("Start training")
        env = self.env
        if self.xinit is None:
            xinit = env.world.sample_xinit()
        else:
            xinit = self.xinit

        world = env.world
        T = env.horizon
        skip = self.skip
        P = self.particles
        dimu = world.dimu

        fast_forward_dynamics = FastForwardDynamics(env, n_parallel=self.n_parallel)

        def rollout(u):
            batch = u.shape[1]
            u_new = expand_actions(u, T, skip)
            x = np.zeros((T + 1, batch, len(xinit)))  # {}
            s = {}
            x[0] = np.tile(xinit.reshape(1, -1), (batch, 1))
            rewards = np.zeros((T, batch))
            for t in range(T):
                x[t + 1], rewards[t], s[t] = fast_forward_dynamics(x[t], u_new[t])
            rewards_new = np.zeros((u.shape[0], u.shape[1]))
            for i in range(u.shape[0] - 1):
                rewards_new[i, :] = np.sum(rewards[(i * skip):((i + 1) * skip), :], 0)
            rewards_new[u.shape[0] - 1, :] = np.sum(rewards[(i * skip):, :], 0)

            if self.time_multiplier is not None:
                rewards_new = self.time_multiplier[:, None] * rewards_new

            return rewards_new, s, x

        # PI2 (notation from https://arxiv.org/pdf/1610.00529v1.pdf)

        if self.init_k is None:
            if self.correlated_noise:
                k = np.zeros((ceildiv(T, skip), dimu))
                k[0] = world.get_stall_action(xinit)
            else:
                k = np.tile(world.get_stall_action(xinit), (ceildiv(T, skip), 1))
        else:
            k = self.init_k

        best_particle = np.copy(k)
        init_cov = self.init_cov
        if isinstance(init_cov, np.ndarray) and len(init_cov.shape) == 1:
            init_cov = np.diag(init_cov)[None, :, :]
        K = init_cov * np.tile(np.eye(dimu).reshape(1, dimu, dimu), (ceildiv(T, skip), 1, 1))
        new_k = np.copy(k)
        new_K = np.copy(K)

        def preprocess_actions(actions):
            if not self.correlated_noise:
                return actions
            result = np.copy(actions)
            for t in range(1, ceildiv(T, skip)):
                result[t] += result[t - 1]
            return result

        if self.num_iterations == -1:
            k = np.random.randn(*k.shape)

        for itr in range(self.num_iterations):
            logger.log("Start iteration #{0}".format(itr))
            # sample
            samples = np.empty((ceildiv(T, skip), P, dimu))
            for t in range(ceildiv(T, skip)):
                # logger.debug('mean shape: %s, cov shape: %s', k[t].shape, K[t].shape)
                samples[t] = np.random.multivariate_normal(k[t], K[t], (P,))
            # if itr >= 1:
            samples[:, 0, :] = best_particle
            # compute reward
            actions = preprocess_actions(samples)
            logger.log("Collecting trajectories")
            rewards_val, senses, xs = rollout(actions)
            logger.log("Collected")
            S = np.zeros(P)

            logger.log("Computing policy update")
            for t in reversed(range(ceildiv(T, skip))):
                S += rewards_val[t]
                for i in range(P):
                    if math.isnan(S[i]):
                        S[i] = -1E100
                if t == 0:
                    best_particle = samples[:, S.argmax(), :]
                temp = 1E-100 if itr == self.num_iterations - 1 else 1e-3
                while temp < 100:
                    prob = softmax(S / temp)
                    # refit Gaussains
                    new_k[t] = np.sum(prob.reshape(-1, 1) * samples[t], 0)
                    centered = samples[t] - new_k[t].reshape(1, -1)
                    cov = np.matmul(centered.reshape(P, dimu, 1), centered.reshape(P, 1, dimu))
                    assert (cov.shape == (P, dimu, dimu))
                    new_K[t] = np.sum(prob.reshape(-1, 1, 1) * cov, 0)
                    KL = kl_gaussians(k[t], K[t], new_k[t], new_K[t])
                    if KL < self.max_kl or itr == self.num_iterations - 1:
                        break
                    else:
                        temp *= 2.
                k[...] = new_k
                K[...] = new_K
            logger.log("Computed")

            logger.record_tabular('Iteration', itr)
            logger.record_tabular('Temperature', temp)
            logger.record_tabular('CovNorm', np.linalg.norm(K[:-1, :, :]))
            logger.record_tabular_misc_stat('Return', S, placement='front')
            logger.record_tabular_misc_stat('FinalReward', rewards_val[-1], placement='front')
            logger.dump_tabular()
            logger.log("Saving params")
            logger.save_itr_params(
                itr=itr,
                params=dict(env=env, samples=samples, S=S, xinit=xinit),
                use_cloudpickle=True
            )
            logger.log("Saved")
        k = preprocess_actions(k)
        k = expand_actions(k, T, skip)

        return LinearController(env, xinit, k).sample_rollouts()
