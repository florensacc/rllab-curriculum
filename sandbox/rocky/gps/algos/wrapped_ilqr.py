from __future__ import print_function
from __future__ import absolute_import
from rllab.algos.base import RLAlgorithm
import numpy as np
import theano.tensor as TT
from rllab.core.serializable import Serializable
from rllab.misc import ext
from rllab.envs.proxy_env import ProxyEnv
from rllab.envs.base import Step

from sandbox.rocky.gps.imported.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from sandbox.rocky.gps.imported.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from sandbox.rocky.gps.imported.dynamics.dynamics_lr import DynamicsLR
from sandbox.rocky.gps.imported.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from sandbox.rocky.gps.imported.policy.lin_gauss_init import init_pd
from sandbox.rocky.gps.imported.sample_list import SampleList
from rllab.sampler import parallel_sampler
from rllab.misc import logger


class NonstopEnv(ProxyEnv):
    def step(self, action):
        ret = self.wrapped_env.step(action)
        return Step(observation=ret.observation, reward=ret.reward, done=False, **ret.info)


class EnvAgent(Serializable):
    def __init__(self, env, T):
        Serializable.quick_init(self, locals())
        self.env = env
        self.T = T
        self.x0 = env.reset()
        self.dU = self.env.action_dim
        self.dX = self.env.observation_space.flat_dim
        self.dO = 0
        self.dM = 0

    def sample(self, policy, cond=0, verbose=False):
        observations = []
        actions = []
        obs = self.env.reset()
        assert cond == 0
        for t in range(self.T):
            act = policy.act(x=obs, obs=None, t=t, noise=np.random.randn(self.dU))
            observations.append(obs)
            actions.append(act)
            obs = self.env.step(act)[0]
        return SimpleSample(observations, actions)


class EnvCost(object):
    def __init__(self, params):
        self.env = params['env']

        obs_var = TT.vector('obs')
        action_var = TT.vector('action')
        cost_var = -self.env.reward_sym(obs_var, action_var)
        l_var = cost_var
        lx_var = TT.grad(cost_var, obs_var, disconnected_inputs='ignore')
        lu_var = TT.grad(cost_var, action_var, disconnected_inputs='ignore')
        lxx_var = TT.hessian(cost_var, obs_var, disconnected_inputs='ignore')
        luu_var = TT.hessian(cost_var, action_var, disconnected_inputs='ignore')
        lux_var = TT.jacobian(TT.grad(cost_var, action_var, disconnected_inputs='ignore'), obs_var,
                              disconnected_inputs='ignore')
        self.f_vals = ext.compile_function(
            [obs_var, action_var],
            [l_var, lx_var, lu_var, lxx_var, luu_var, lux_var],
            log_name="f_vals"
        )

    def eval(self, sample):
        T = sample.get_X().shape[0]
        Du = sample.get_U().shape[1]
        Dx = sample.get_X().shape[1]

        l = np.zeros(T)
        lx = np.zeros((T, Dx))
        lu = np.zeros((T, Du))
        lxx = np.zeros((T, Dx, Dx))
        luu = np.zeros((T, Du, Du))
        lux = np.zeros((T, Du, Dx))

        for t in range(T):
            x = sample.get_X(t)
            u = sample.get_U(t)
            l[t], lx[t], lu[t], lxx[t], luu[t], lux[t] = self.f_vals(x, u)
        return l, lx, lu, lxx, luu, lux


class PolicyWrapper(object):
    def __init__(self, gps_policy):
        self.gps_policy = gps_policy
        self.t = None

    def reset(self):
        self.t = 0

    def get_action(self, observation):
        action = self.gps_policy.act(x=observation, obs=None, t=self.t, noise=np.random.randn(self.gps_policy.dU))
        self.t += 1
        return action, dict()

    def get_param_values(self):
        return self.gps_policy

    def set_param_values(self, val):
        self.gps_policy = val


class SimpleSample(object):
    def __init__(self, observations, actions):
        self.observations = np.asarray(observations)
        self.actions = np.asarray(actions)

    def get_X(self, t=None):
        if t is None:
            return self.observations
        return self.observations[t]

    def get_U(self, t=None):
        if t is None:
            return self.actions
        return self.actions[t]


class WrappedILQR(RLAlgorithm):
    """
    A wrapper for the iLQR implementation in the GPS codebase under unknown dynamics
    """

    def __init__(
            self,
            env,
            n_itr=100,
            horizon=100,
            n_paths=20,
            init_controller_std=1.0):
        """
        :param env:
        :param n_itr: Number of training iterations
        :param horizon: Length of a single trajectory. In GPS, it is assumed that all trajectories will have the same
                        length
        :param n_paths: Number of sample trajectories per iteration
        :param init_controller_std: Standard deviation of the initial controller noise
        """
        self.env = env
        self.n_itr = n_itr
        self.horizon = horizon
        self.n_paths = n_paths
        self.init_controller_std = init_controller_std

    def train(self):
        agent = EnvAgent(env=self.env, T=self.horizon)

        algo = AlgorithmTrajOpt(dict(
            conditions=1,
            agent=agent,
            traj_opt=dict(
                type=TrajOptLQRPython,
            ),
            cost=dict(
                type=EnvCost,
                env=self.env,
                T=self.horizon,
            ),
            dynamics=dict(
                type=DynamicsLR,
                regularization=1e-6,
                prior=dict(
                    type=DynamicsPriorGMM,
                    max_clusters=20,
                    min_samples_per_cluster=40,
                    max_samples=20,
                )
            ),
            init_traj_distr=dict(
                type=init_pd,
                init_var=self.init_controller_std ** 2,
                pos_gains=0.0,
                dQ=self.env.action_dim,
                T=self.horizon,
            )
        ))

        policy_wrapper = PolicyWrapper(algo.cur[0].traj_distr)
        parallel_sampler.populate_task(NonstopEnv(self.env), policy_wrapper)

        for itr in xrange(self.n_itr):
            policy = algo.cur[0].traj_distr
            paths = parallel_sampler.sample_paths(
                policy,
                max_samples=self.horizon * self.n_paths,
                max_path_length=self.horizon,
            )
            paths = paths[:self.n_paths]
            sample_list = SampleList([SimpleSample(p["observations"], p["actions"]) for p in paths])
            algo.iteration(sample_lists=[sample_list])
            average_return = np.mean([np.sum(p["rewards"]) for p in paths])
            logger.record_tabular("Itr", itr)
            logger.record_tabular("AverageReturn", average_return)
            logger.dump_tabular()
