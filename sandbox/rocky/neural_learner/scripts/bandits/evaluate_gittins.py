import multiprocessing
import os
import tempfile

from rllab import config
from sandbox.rocky.neural_learner.envs.mab_env import MABEnv
from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv

import numpy as np
import numba
import csv
import scipy.stats

"""Computes an approximated Gittins Index, using backward induction for a fixed horizon length.

For more see Gittins etal. 2011 MAB Allocation Indices, 2nd ed.
For example the authors list precomputed index values for some parameter specifications on p.265.
"""

import numpy as np
import time

__author__ = 'Ilari'

folder_name = "iclr2016_new"


def initial_approximation(pulls, discount, grid_n):
    """Approximate the initial values for the value function to begin backward induction.

    Pulls specifies the total number of bandit arm pulls and observations from which backward
    induction is used to compute the index values for any distribution of discrete binary
    observations. Success denoted by a, and failure denoted by b.

    Assumptions 1 <= a,b <= pulls - 1, so we assume at least one observation of success and failure.

    :param pulls: scalar
    :param discount: Discount factor from interval (0,1)
    :param grid_n

    :return gittins, values: Initialized index array and value array
    """

    values = np.zeros([pulls - 1, pulls - 1, grid_n])  # Store V(a=k, b=n-k, r) in values[k,n-1,:] as k varies
    gittins = np.zeros([pulls - 1, pulls - 1])  # Store Gittins(a=k, b=n-k) in gittins[k,n-1] as k varies

    a_grid = np.arange(1, pulls)
    r_grid = np.linspace(0, 1, grid_n)

    initial_gittins = a_grid / float(pulls)  # Initial Gittins Approximation to start Backward Induction
    gittins[0:pulls, pulls - 2] = initial_gittins  # Record initial Gittins approximation

    for idx_a, a in enumerate(a_grid):
        values[idx_a, pulls - 2, :] = (1.0 / (1 - discount)) * \
                                      np.maximum(r_grid, a / float(pulls))  # Record initial Value approximation

    return gittins, values


def recursion_step(value_n, r_grid, discount):
    """One-step backward induction computing the value function and the Gittins Index.

     See for instance Gittins etal 2011, or Powell and Ryzhov 2012, for recursion step details.
     """

    n = value_n.shape[0]
    r_len = r_grid.shape[0]
    value_n_minus_1 = np.zeros([n - 1, r_len])  # Value function length reduced by 1
    gittins_n_minus_1 = np.zeros(n - 1)  # Value function length reduced by 1
    for k in range(0, n - 1):
        a = k + 1  # a in range [1,n-1]
        b = n - k - 1  # b in range [1,n-1]
        value_n_minus_1[k, :] = np.maximum((r_grid / float(1 - discount)),
                                           (a / float(n)) * (1 + discount * value_n[k + 1, :]) +
                                           (b / float(n)) * discount * value_n[k, :]
                                           )
        try:
            # Find first index where Value = (Value of Safe Arm)
            idx_git = np.argwhere((r_grid / float(1 - discount)) == value_n_minus_1[k, :]).flatten()
            gittins_n_minus_1[k] = 0.5 * (r_grid[idx_git[0]] + r_grid[idx_git[0] - 1])  # Take average
        except:
            print("Error in finding Gittins index")

    return gittins_n_minus_1, value_n_minus_1


def recursion_loop(pulls, discount, grid_n):
    """This produces the value functions and Gittins indices by backward induction"""

    r_grid = np.linspace(0, 1, grid_n)
    gittins, values = initial_approximation(pulls, discount, grid_n)
    n = pulls - 2  # Note that the 2 comes from (1) the initial approximation and (2) python indexing
    while n >= 1:
        g, v = recursion_step(values[:n + 1, n, :], r_grid, discount)
        values[:n, n - 1] = v
        gittins[:n, n - 1] = g
        n -= 1
    return gittins, values


def reformat_gittins(g, v=None):
    """Reformat output.

    We reformat the result to get the results in a similar form
    as in (Gittins etal 2011, Powell and Ryzhov 2012), except that we store:
    Success count denoted by a in rows
    Failure count denoted by b in columns
    """

    start_time = time.time()
    n = g.shape[0]
    g_reformat = np.zeros(g.shape)

    for row in range(0, n):
        g_reformat[row, :n - row] = g[row, row:]

    # try:
    #     v_reformat = np.zeros(v.shape)
    #     for row in range(0, n):
    #         v_reformat[row, :n - row, :] = v[row, row:, :]
    #     print("Elapsed time in Gittins Index Reformatting: ", time.time() - start_time)
    #     return g_reformat, v_reformat
    # except:
    print("Elapsed time in Gittins Index Reformatting: ", time.time() - start_time)
    return g_reformat


def gittins_index(n=500, grid=1000, discount=0.9, value=False, df=False):
    """Compute Gittins indices and value functions.

    Comment: To get the results to match up, with See Gittins etal. (2011, p.265)
    we need a fairly fine grid: approx 5000 grid points, equates the results.

    :param n: Number of pulls, from which to start backward induction
    :param grid: Number of grid points to use for safe arm
    :param discount: discount factor used to compute Gittins Index
    :param value: If True, function return value functions, in addition to Gittins Index
    :param df: If True, function returns a Pandas DataFrame of the results in addition to Numpy Arrays

    :return g: Gittins index, (n x n) array
               rows: a count, i.e. number of successes.
               columns: b count, i.e. number of failures.
     :return v: Value function, (n x n x grid) array
                rows: a count, i.e. number of successes.
                columns: b count, i.e. number of failures.
                dimension 3: r grid for the bernoulli parameter of the certain arm.
    """

    print("Computing Gittins Index...")
    start_time = time.time()
    g, v = recursion_loop(n, discount, grid)
    print("Elapsed time in Gittins Index Calculation: ", time.time() - start_time)
    g_reformat = reformat_gittins(g)  # , v)
    return g_reformat


from sandbox.rocky.s3.resource_manager import resource_manager


def gen_gittins(n, grid, discount):
    resource_name = "gittins/n_%d_grid_%d_discount_%f.pkl" % (n, grid, discount)

    def mk():
        data = gittins_index(n=n, grid=grid, discount=discount)
        f = tempfile.NamedTemporaryFile()
        f_name = f.name + ".npz"
        np.savez_compressed(f_name, data=data)
        print("uploading...")
        resource_manager.register_file(resource_name, f_name)

    f = resource_manager.get_file(resource_name, mk)
    return np.load(f)["data"]


n = 1000
grid = 5000

GITTINS = dict()
for discount in [0.9, 0.99, 0.999, 0.9999]:
    GITTINS[discount] = gen_gittins(n, grid, discount)

# def memoize(function):
#     memo = {}
#
#     def wrapper(*args):
#         if args in memo:
#             return memo[args]
#         else:
#             rv = function(*args)
#             memo[args] = rv
#             return rv
#
#     return wrapper
#
#
# def precompute_gittins(discount, n)
#
#
# @memoize
# @numba.njit
# def _compute_arm_index(a, b, max_k, discount=1.0):
#     nu = 0.0
#     best_k = 0
#     v = np.zeros(max_k)
#     for k in range(max_k):
#         for j in range(k + 1):
#             v[j] = (a + j) / (a + k + b)
#         for i in range(k - 1, -1, -1):
#             for j in range(i + 1):
#                 v[j] = (a + j) / (a + i + b) * (1.0 + discount * v[j + 1]) + (b + i - j) / (a + i + b) * discount * \
#                                                                              v[j]
#         vk = v[0] / (k + 1.0)
#         if vk >= nu:
#             nu = vk
#             best_k = k + 1
#     return nu, best_k
#
#
class ExactGittinsBernoulli(object):
    def __init__(self, max_k=500):
        self.num_arms = 0
        self.alphas = None
        self.betas = None
        self.max_k = max_k

    def set_parameters(self, num_arms):
        if num_arms < 2:
            raise ValueError("number of arms must be >= 2")

        self.num_arms = num_arms
        self.alphas = np.ones(self.num_arms, dtype=np.int)
        self.betas = np.ones(self.num_arms, dtype=np.int)

    def make_choice(self):  # , horizon):
        # import ipdb; ipdb.set_trace()
        # if np.sum(self.betas)+np.sum(self.alphas) < 3*self.num_arms:
        #     for i in range(self.num_arms):
        #         if self.betas[i]+self.alphas[i] < 3:
        #             return i

        return np.argmax(GITTINS[discount][self.alphas, self.betas])

        # indices = np.zeros(self.num_arms)
        # for i in range(self.num_arms):
        #     nu, best_k = _compute_arm_index(self.alphas[i], self.betas[i], self.max_k)  # min(self.max_k, horizon))
        #     indices[i] = nu
        #
        # return np.argmax(indices)

    # def make_choice(self):
    #     indices = np.zeros(self.num_arms)
    #     for i in range(self.num_arms):
    #         nu = _compute_arm_index(self.alphas[i], self.betas[i], 1.)
    #         indices[i] = nu
    #
    #     return np.argmax(indices)

    def save_outcome(self, index, success):
        if success:
            self.alphas[index] += 1
        else:
            self.betas[index] += 1

    def get_means(self):
        return self.alphas / (self.alphas + self.betas)


def evaluate_gittins_once(env, seed=None, discount=0.99):
    np.random.seed(seed)
    env.reset()

    policy = ExactGittinsBernoulli()
    policy.set_parameters(num_arms=env.action_space.flat_dim)

    rewards = []

    for t in range(env.n_episodes):
        action = policy.make_choice()
        _, reward, _, _ = env.step(action)
        policy.save_outcome(index=action, success=reward)
        rewards.append(reward)

    return np.sum(rewards)


def evaluate_gittins():
    write_file = os.path.join(config.PROJECT_PATH, "data/%s/gittins_mab.csv" % folder_name)
    with open(write_file, "w") as f:

        writer = csv.DictWriter(f, ["n_arms", "n_episodes", "avg", "stdev", "best_discount"])
        writer.writeheader()

        for arms in [5, 10, 50]:

            for n_episodes in [10, 100, 500]:
                env = MultiEnv(wrapped_env=MABEnv(n_arms=arms), n_episodes=n_episodes, episode_horizon=1, discount=1)
                best_mean = None
                best_results = None

                for discount in [0.9, 0.99, 0.999, 0.9999]:

                    with multiprocessing.Pool() as pool:
                        returns = pool.starmap(evaluate_gittins_once, [(env, seed, discount) for seed in range(1000)])

                    mean = np.mean(returns)
                    std = np.std(returns) / np.sqrt(len(returns) - 1)
                    results = (discount, mean, std)

                    if best_mean is None or mean > best_mean:
                        best_mean = mean
                        best_results = results

                    print(results)

                writer.writerow(dict(
                    n_arms=arms,
                    n_episodes=n_episodes,
                    avg=best_results[1],
                    stdev=best_results[2],
                    best_discount=best_results[0],
                ))

                f.flush()

                print("Arms:", arms, flush=True)
                print("N episodes:", n_episodes, flush=True)
                print("Average return:", best_results[1], flush=True)


if __name__ == "__main__":
    evaluate_gittins()
