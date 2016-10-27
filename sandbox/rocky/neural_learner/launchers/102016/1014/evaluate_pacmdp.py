from sandbox.rocky.neural_learner.TabulaRL.src.environment import TabularMDP
from sandbox.rocky.neural_learner.TabulaRL.src.experiment import run_finite_tabular_experiment
from sandbox.rocky.neural_learner.TabulaRL.src.feature_extractor import FeatureTrueState
from sandbox.rocky.neural_learner.TabulaRL.src.finite_tabular_agents import PSRL
from sandbox.rocky.neural_learner.envs.random_tabular_mdp_env import RandomTabularMDPEnv
import numpy as np



class TabularMDPBridge(TabularMDP):

    def __init__(self, nState, nAction, epLen):
        super().__init__(nState=nState, nAction=nAction, epLen=epLen)
        self.env = RandomTabularMDPEnv(n_states=nState, n_actions=nAction)
        self.R = np.zeros((nState, nAction, 2))
        self.P = self.env.executor.Ps[0]
        self.R[:, :, 0] = self.env.executor.Rs[0]
        self.R[:, :, 1] = 1. / np.sqrt(self.env.tau)
    #     self.timestep = 0
    #
    # def reset(self):
    #     state = self.env.reset()
    #     self.state = state
    #     self.timestep = 0
    #     return self.state
    #
    # def advance(self, action):
    #     next_state, reward, done, _ = self.env.step(action)
    #     self.timestep += 1
    #     done = done or self.timestep >= self.epLen
    #     if done:
    #         self.reset()
    #     return reward, next_state, done


episode_horizon = 10
n_states = 10
n_actions = 5
n_episodes = 50



returns = []
for idx in range(2500):
    env = TabularMDPBridge(nState=n_states, nAction=n_actions, epLen=episode_horizon)
    agent = PSRL(nState=n_states, nAction=n_actions, epLen=episode_horizon)

    returns.append(run_finite_tabular_experiment(agent, env, FeatureTrueState(env.epLen, env.nState, env.nAction,
                                                                             env.nState), nEps=n_episodes, seed=idx))
    print(idx, np.mean(returns))