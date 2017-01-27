import csv
import multiprocessing
import os
import pickle

from rllab import config
from rllab.core.serializable import Serializable
from sandbox.rocky.neural_learner.TabulaRL.src.environment import TabularMDP
from sandbox.rocky.neural_learner.TabulaRL.src.experiment import run_finite_tabular_experiment
from sandbox.rocky.neural_learner.TabulaRL.src.feature_extractor import FeatureTrueState
from sandbox.rocky.neural_learner.TabulaRL.src.finite_tabular_agents import PSRL, OptimisticPSRL, BEB, EpsilonGreedy, \
    UCRL2
from sandbox.rocky.neural_learner.envs.random_tabular_mdp_env import RandomTabularMDPEnv
import numpy as np

CFGs = [
    (10, 5, 10, 10),
]


class TabularMDPBridge(Serializable):
    def __init__(self, nState, nAction, epLen, nEp, seed):
        Serializable.quick_init(self, locals())
        import gym
        self.env = gym.make(
            'RandomTabularMDP-{s}.states-{a}.actions-{t}.timesteps-{n}.episodes-v0'.format(
                s=nState, a=nAction, t=epLen, n=nEp
            )
        )
        self.env.seed(seed)
        self.env.reset()
        self.epLen = epLen
        self.nState = nState
        self.nAction = nAction
        self.tab_env = TabularMDP(nState=nState, nAction=nAction, epLen=epLen)
        self.tab_env.R = {}
        self.tab_env.P = {}
        for state in range(nState):
            for action in range(nAction):
                self.tab_env.R[state, action] = (self.env._R[state, action], 1)
                self.tab_env.P[state, action] = self.env._P[state, action]  # np.ones(nState) / nState
                # import ipdb; ipdb.set_trace()

    def compute_qVals(self):
        return self.tab_env.compute_qVals()

    @property
    def timestep(self):
        return self.env._episode_t

    @property
    def state(self):
        return self.env._current_state

    def advance(self, action):
        obs, rew, done, _ = self.env.step(action)
        pContinue = self.env._episode_t != 0
        return rew, self.state, pContinue#(self.env._episode_t == 0)

    def reset(self):
        pass
        # self.env.reset()
        # self.tab_env.R = {}
        # self.R = np.zeros((nState, nAction, 2))
        # self.P = self.env.executor.Ps[0]
        # self.R[:, :, 0] = self.env.executor.Rs[0]
        # self.R[:, :, 1] = 1. / np.sqrt(self.env.tau)


def evaluate_psrl_once(cfg, seed):
    n_states, n_actions, episode_horizon, n_episodes = cfg
    env = TabularMDPBridge(nState=n_states, nAction=n_actions, epLen=episode_horizon, nEp=n_episodes, seed=seed)
    agent = PSRL(nState=n_states, nAction=n_actions, epLen=episode_horizon)
    return run_finite_tabular_experiment(
        agent, env,
        FeatureTrueState(
            env.epLen, env.nState, env.nAction, env.nState
        ),
        nEps=n_episodes,
        seed=seed
    )
#
#
# evaluate_psrl_once(CFGs[0], seed=0)


def evaluate_psrl():

    n_trials = 100#0

    for cfg in CFGs:
        n_states, n_actions, episode_horizon, n_episodes = cfg
        with multiprocessing.Pool() as pool:
            returns = pool.starmap(evaluate_psrl_once, [(cfg, seed) for seed in range(n_trials)])

        # import ipdb; ipdb.set_trace()

        print(dict(
            n_states=n_states,
            n_actions=n_actions,
            episode_horizon=episode_horizon,
            n_episodes=n_episodes,
            avg=np.mean(returns),
            stdev=np.std(returns) / np.sqrt(n_trials - 1)
        ))


if __name__ == "__main__":
    evaluate_psrl()
