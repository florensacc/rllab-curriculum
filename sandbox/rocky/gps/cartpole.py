from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.gps.envs.symbolic_swimmer_env import SymbolicSwimmerEnv
from sandbox.rocky.gps.imported.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from sandbox.rocky.gps.imported.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from sandbox.rocky.gps.imported.dynamics.dynamics_lr import DynamicsLR
from sandbox.rocky.gps.imported.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from sandbox.rocky.gps.imported.policy.lin_gauss_init import init_pd
from sandbox.rocky.gps.imported.sample_list import SampleList
from rllab.misc import ext
from rllab.sampler.utils import rollout
import numpy as np
import theano.tensor as TT
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

env = SymbolicSwimmerEnv()


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


class EnvAgent(object):
    def __init__(self, env, T):
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

        # self._worlds[condition].run()
        # self._worlds[condition].reset_world()
        # b2d_X = self._worlds[condition].get_state()
        # new_sample = self._init_sample(b2d_X)
        # U = np.zeros([self.T, self.dU])
        # noise = generate_noise(self.T, self.dU, self._hyperparams)
        # for t in range(self.T):
        #     X_t = new_sample.get_X(t=t)
        #     obs_t = new_sample.get_obs(t=t)
        #     U[t, :] = policy.act(X_t, obs_t, t, noise[t, :])
        #     if (t+1) < self.T:
        #         for _ in range(self._hyperparams['substeps']):
        #             self._worlds[condition].run_next(U[t, :])
        #         b2d_X = self._worlds[condition].get_state()
        #         self._set_sample(new_sample, b2d_X, t)
        # new_sample.set(ACTION, U)
        # if save:
        #     self._samples[condition].append(new_sample)
        # import ipdb; ipdb.set_trace()


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


class EnvCost(object):
    def __init__(self, params):
        env = params['env']
        # assert isinstance(env, SymbolicCartpoleEnv)
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


agent = EnvAgent(env, T=500)

num_samples = 20#20#5

algo = AlgorithmTrajOpt(dict(
    conditions=1,
    agent=agent,
    traj_opt=dict(
        type=TrajOptLQRPython,
    ),
    cost=dict(
        type=EnvCost,
        env=env,
        T=500,
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
        init_var=50.**2/9,
        pos_gains=0.0,
        dQ=env.action_dim,
        T=500,
    )
))

for itr in xrange(100):
    policy = algo.cur[0].traj_distr
    sample_list = []
    for _ in xrange(num_samples):
        sample_list.append(agent.sample(policy, cond=0, verbose=True))
    algo.iteration(sample_lists=[SampleList(sample_list)])
    path = rollout(env, PolicyWrapper(policy), max_path_length=500, animated=True)
    print("Actual return: %f" % np.sum(path["rewards"]))
