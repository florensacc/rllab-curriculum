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
    (10, 5, 10, 25),
    (10, 5, 10, 50),
    (10, 5, 10, 75),
    (10, 5, 10, 100),
]

folder_name = "iclr2016_full"


class TabularMDPBridge(TabularMDP, Serializable):
    def __init__(self, nState, nAction, epLen):
        Serializable.quick_init(self, locals())
        super().__init__(nState=nState, nAction=nAction, epLen=epLen)
        self.env = RandomTabularMDPEnv(n_states=nState, n_actions=nAction)
        self.R = np.zeros((nState, nAction, 2))
        self.P = self.env.executor.Ps[0]
        self.R[:, :, 0] = self.env.executor.Rs[0]
        self.R[:, :, 1] = 1. / np.sqrt(self.env.tau)


def evaluate_psrl_once(cfg, seed):
    n_states, n_actions, episode_horizon, n_episodes = cfg
    np.random.seed(seed)
    env = TabularMDPBridge(nState=n_states, nAction=n_actions, epLen=episode_horizon)
    agent = PSRL(nState=n_states, nAction=n_actions, epLen=episode_horizon)
    return run_finite_tabular_experiment(
        agent, env,
        FeatureTrueState(
            env.epLen, env.nState, env.nAction, env.nState
        ),
        nEps=n_episodes,
        seed=seed
    )


def evaluate_optimistic_psrl_once(cfg, n_samp, seed):
    n_states, n_actions, episode_horizon, n_episodes = cfg
    np.random.seed(seed)
    env = TabularMDPBridge(nState=n_states, nAction=n_actions, epLen=episode_horizon)
    agent = OptimisticPSRL(nState=n_states, nAction=n_actions, epLen=episode_horizon, nSamp=n_samp)
    return run_finite_tabular_experiment(
        agent, env,
        FeatureTrueState(
            env.epLen, env.nState, env.nAction, env.nState
        ),
        nEps=n_episodes,
        seed=seed
    )


def evaluate_beb_once(cfg, scaling, seed):
    n_states, n_actions, episode_horizon, n_episodes = cfg
    np.random.seed(seed)
    env = TabularMDPBridge(nState=n_states, nAction=n_actions, epLen=episode_horizon)
    agent = BEB(nState=n_states, nAction=n_actions, epLen=episode_horizon, scaling=scaling)
    return run_finite_tabular_experiment(
        agent, env,
        FeatureTrueState(
            env.epLen, env.nState, env.nAction, env.nState
        ),
        nEps=n_episodes,
        seed=seed
    )


def evaluate_ucrl2_once(cfg, scaling, seed):
    n_states, n_actions, episode_horizon, n_episodes = cfg
    np.random.seed(seed)
    env = TabularMDPBridge(nState=n_states, nAction=n_actions, epLen=episode_horizon)
    agent = UCRL2(nState=n_states, nAction=n_actions, epLen=episode_horizon, scaling=scaling)
    return run_finite_tabular_experiment(
        agent, env,
        FeatureTrueState(
            env.epLen, env.nState, env.nAction, env.nState
        ),
        nEps=n_episodes,
        seed=seed
    )


def evaluate_greedy_once(cfg, seed):
    n_states, n_actions, episode_horizon, n_episodes = cfg
    np.random.seed(seed)
    env = TabularMDPBridge(nState=n_states, nAction=n_actions, epLen=episode_horizon)
    agent = EpsilonGreedy(nState=n_states, nAction=n_actions, epLen=episode_horizon, epsilon=0.)
    return run_finite_tabular_experiment(
        agent, env,
        FeatureTrueState(
            env.epLen, env.nState, env.nAction, env.nState
        ),
        nEps=n_episodes,
        seed=seed
    )


def evaluate_epsilon_greedy_once(cfg, epsilon, seed):
    n_states, n_actions, episode_horizon, n_episodes = cfg
    np.random.seed(seed)
    env = TabularMDPBridge(nState=n_states, nAction=n_actions, epLen=episode_horizon)
    agent = EpsilonGreedy(nState=n_states, nAction=n_actions, epLen=episode_horizon, epsilon=epsilon)
    return run_finite_tabular_experiment(
        agent, env,
        FeatureTrueState(
            env.epLen, env.nState, env.nAction, env.nState
        ),
        nEps=n_episodes,
        seed=seed
    )


def evaluate_random_once(cfg, seed):
    n_states, n_actions, episode_horizon, n_episodes = cfg
    np.random.seed(seed)
    env = TabularMDPBridge(nState=n_states, nAction=n_actions, epLen=episode_horizon)
    agent = EpsilonGreedy(nState=n_states, nAction=n_actions, epLen=episode_horizon, epsilon=1.)
    return run_finite_tabular_experiment(
        agent, env,
        FeatureTrueState(
            env.epLen, env.nState, env.nAction, env.nState
        ),
        nEps=n_episodes,
        seed=seed
    )


def evaluate_psrl():
    write_file = os.path.join(config.PROJECT_PATH, "data/%s/psrl_mdp.csv" % folder_name)

    n_trials = 1000

    with open(write_file, "w") as f:
        writer = csv.DictWriter(f, ["n_states", "n_actions", "episode_horizon", "n_episodes", "avg", "stdev"])
        writer.writeheader()

        for cfg in CFGs:
            n_states, n_actions, episode_horizon, n_episodes = cfg
            with multiprocessing.Pool() as pool:
                returns = pool.starmap(evaluate_psrl_once, [(cfg, seed) for seed in range(1000)])

            writer.writerow(dict(
                n_states=n_states,
                n_actions=n_actions,
                episode_horizon=episode_horizon,
                n_episodes=n_episodes,
                avg=np.mean(returns),
                stdev=np.std(returns) / np.sqrt(n_trials - 1)
            ))
            f.flush()


def evaluate_optimistic_psrl():
    write_file = os.path.join(config.PROJECT_PATH, "data/%s/optimistic_psrl_mdp.csv" % folder_name)

    n_trials = 1000

    with open(write_file, "w") as f:
        writer = csv.DictWriter(f, ["n_states", "n_actions", "episode_horizon", "n_episodes", "avg", "stdev",
                                    "best_n_samp"])
        writer.writeheader()

        for cfg in CFGs:

            n_states, n_actions, episode_horizon, n_episodes = cfg

            best_result = None
            best_mean = None

            for n_samp in range(1, 21):
                with multiprocessing.Pool() as pool:
                    returns = pool.starmap(evaluate_optimistic_psrl_once, [(cfg, n_samp, seed) for seed in range(1000)])
                    result = dict(
                        n_states=n_states,
                        n_actions=n_actions,
                        episode_horizon=episode_horizon,
                        n_episodes=n_episodes,
                        avg=np.mean(returns),
                        stdev=np.std(returns) / np.sqrt(n_trials - 1),
                        best_n_samp=n_samp
                    )
                    if best_mean is None or result["avg"] > best_mean:
                        best_result = result
                        best_mean = result["avg"]

                    print(result)

            writer.writerow(best_result)
            f.flush()


def evaluate_beb():
    write_file = os.path.join(config.PROJECT_PATH, "data/%s/beb_mdp.csv" % folder_name)

    n_trials = 1000

    with open(write_file, "w") as f:
        writer = csv.DictWriter(f, ["n_states", "n_actions", "episode_horizon", "n_episodes", "avg", "stdev",
                                    "best_scaling"])
        writer.writeheader()

        for cfg in CFGs:

            n_states, n_actions, episode_horizon, n_episodes = cfg

            best_result = None
            best_mean = None

            scalings = np.exp(np.linspace(np.log(0.0001), np.log(1.0), 21))

            # scalings = np.log(0.001)

            for scaling in scalings:  # [x * 0.1 for x in range(1, 21)]:
                with multiprocessing.Pool() as pool:
                    returns = pool.starmap(evaluate_beb_once, [(cfg, scaling, seed) for seed in range(1000)])
                    result = dict(
                        n_states=n_states,
                        n_actions=n_actions,
                        episode_horizon=episode_horizon,
                        n_episodes=n_episodes,
                        avg=np.mean(returns),
                        stdev=np.std(returns) / np.sqrt(n_trials - 1),
                        best_scaling=scaling
                    )
                    if best_mean is None or result["avg"] > best_mean:
                        best_result = result
                        best_mean = result["avg"]

                    print(result)

            writer.writerow(best_result)
            f.flush()


def evaluate_greedy():
    write_file = os.path.join(config.PROJECT_PATH, "data/%s/greedy_mdp.csv" % folder_name)

    n_trials = 1000

    with open(write_file, "w") as f:
        writer = csv.DictWriter(f, ["n_states", "n_actions", "episode_horizon", "n_episodes", "avg", "stdev"])
        writer.writeheader()

        for cfg in CFGs:
            n_states, n_actions, episode_horizon, n_episodes = cfg

            with multiprocessing.Pool() as pool:
                returns = pool.starmap(evaluate_greedy_once, [(cfg, seed) for seed in range(1000)])
                result = dict(
                    n_states=n_states,
                    n_actions=n_actions,
                    episode_horizon=episode_horizon,
                    n_episodes=n_episodes,
                    avg=np.mean(returns),
                    stdev=np.std(returns) / np.sqrt(n_trials - 1),
                )

            writer.writerow(result)
            f.flush()


def evaluate_epsilon_greedy():
    write_file = os.path.join(config.PROJECT_PATH, "data/%s/epsilon_greedy_mdp.csv" % folder_name)

    n_trials = 1000

    with open(write_file, "w") as f:
        writer = csv.DictWriter(f, ["n_states", "n_actions", "episode_horizon", "n_episodes", "avg", "stdev",
                                    "best_epsilon"])
        writer.writeheader()

        for cfg in CFGs:

            n_states, n_actions, episode_horizon, n_episodes = cfg

            best_result = None
            best_mean = None

            for epsilon in [x * 0.1 for x in range(1, 11)]:
                with multiprocessing.Pool() as pool:
                    returns = pool.starmap(evaluate_epsilon_greedy_once, [(cfg, epsilon, seed) for seed in range(1000)])
                    result = dict(
                        n_states=n_states,
                        n_actions=n_actions,
                        episode_horizon=episode_horizon,
                        n_episodes=n_episodes,
                        avg=np.mean(returns),
                        stdev=np.std(returns) / np.sqrt(n_trials - 1),
                        best_epsilon=epsilon
                    )
                    if best_mean is None or result["avg"] > best_mean:
                        best_result = result
                        best_mean = result["avg"]

                    print(result)

            writer.writerow(best_result)
            f.flush()


def evaluate_random():
    write_file = os.path.join(config.PROJECT_PATH, "data/%s/random_mdp.csv" % folder_name)

    n_trials = 1000

    with open(write_file, "w") as f:
        writer = csv.DictWriter(f, ["n_states", "n_actions", "episode_horizon", "n_episodes", "avg", "stdev"])
        writer.writeheader()

        for cfg in CFGs:
            n_states, n_actions, episode_horizon, n_episodes = cfg

            with multiprocessing.Pool() as pool:
                returns = pool.starmap(evaluate_random_once, [(cfg, seed) for seed in range(1000)])
                result = dict(
                    n_states=n_states,
                    n_actions=n_actions,
                    episode_horizon=episode_horizon,
                    n_episodes=n_episodes,
                    avg=np.mean(returns),
                    stdev=np.std(returns) / np.sqrt(n_trials - 1),
                )

            writer.writerow(result)
            f.flush()


def evaluate_ucrl2():
    write_file = os.path.join(config.PROJECT_PATH, "data/%s/ucrl2_mdp.csv" % folder_name)

    n_trials = 1000

    with open(write_file, "w") as f:
        writer = csv.DictWriter(f, ["n_states", "n_actions", "episode_horizon", "n_episodes", "avg", "stdev",
                                    "best_scaling"])
        writer.writeheader()

        for cfg in CFGs:

            n_states, n_actions, episode_horizon, n_episodes = cfg

            best_result = None
            best_mean = None

            scalings = np.exp(np.linspace(np.log(0.0001), np.log(1.0), 21))

            # scalings = np.log(0.001)

            for scaling in scalings:  # [x * 0.1 for x in range(1, 21)]:
                with multiprocessing.Pool() as pool:
                    returns = pool.starmap(evaluate_ucrl2_once, [(cfg, scaling, seed) for seed in range(1000)])
                    result = dict(
                        n_states=n_states,
                        n_actions=n_actions,
                        episode_horizon=episode_horizon,
                        n_episodes=n_episodes,
                        avg=np.mean(returns),
                        stdev=np.std(returns) / np.sqrt(n_trials - 1),
                        best_scaling=scaling
                    )
                    if best_mean is None or result["avg"] > best_mean:
                        best_result = result
                        best_mean = result["avg"]

                    print(result)

            writer.writerow(best_result)
            f.flush()


if __name__ == "__main__":
    evaluate_psrl()
    evaluate_optimistic_psrl()
    evaluate_beb()
    evaluate_greedy()
    evaluate_epsilon_greedy()
    evaluate_random()
    evaluate_ucrl2()
