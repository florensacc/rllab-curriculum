import csv
import itertools

from rllab import config
import os
import joblib
import tensorflow as tf
import numpy as np

from rllab.misc import tensor_utils
from rllab.misc.console import mkdir_p
from rllab.sampler.stateful_pool import ProgBarCounter
from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv
from sandbox.rocky.neural_learner.scripts.bandits.evaluate_bandits_split_difficulty import EpsilonMABEnv


def evaluate(env, policy, n_envs, max_path_length):
    np.random.seed(0)
    paths = []
    n_samples = 0
    dones = np.asarray([True] * n_envs)
    vec_env = env.vec_env_executor(n_envs)
    obses = vec_env.reset(dones, seeds=np.arange(n_envs))
    running_paths = [None] * n_envs

    pbar = ProgBarCounter(n_envs * max_path_length)
    policy_time = 0
    env_time = 0
    process_time = 0

    env_spec = env.spec

    import time

    while len(paths) < n_envs:
        t = time.time()
        policy.reset(dones)
        actions, agent_infos = policy.get_actions(obses)

        policy_time += time.time() - t
        t = time.time()
        next_obses, rewards, dones, env_infos = vec_env.step(actions, max_path_length=max_path_length)
        env_time += time.time() - t

        t = time.time()

        agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
        env_infos = tensor_utils.split_tensor_dict_list(env_infos)
        if env_infos is None:
            env_infos = [dict() for _ in range(n_envs)]
        if agent_infos is None:
            agent_infos = [dict() for _ in range(n_envs)]
        for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                rewards, env_infos, agent_infos,
                                                                                dones):
            if running_paths[idx] is None:
                running_paths[idx] = dict(
                    observations=[],
                    actions=[],
                    rewards=[],
                    env_infos=[],
                    agent_infos=[],
                )
            running_paths[idx]["observations"].append(observation)
            running_paths[idx]["actions"].append(action)
            running_paths[idx]["rewards"].append(reward)
            running_paths[idx]["env_infos"].append(env_info)
            running_paths[idx]["agent_infos"].append(agent_info)
            if done:
                paths.append(dict(
                    observations=env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                    actions=env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                    rewards=tensor_utils.stack_tensor_list(running_paths[idx]["rewards"]),
                    env_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                    agent_infos=tensor_utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                ))

                n_samples += len(running_paths[idx]["rewards"])
                running_paths[idx] = None

        process_time += time.time() - t
        pbar.inc(len(obses))
        obses = next_obses

    pbar.stop()

    return paths

EPSILONS = [
    (0, 0.01),
    (0.01, 0.05),
    (0.05, 0.1),
    (0.1, 0.3),
    (0.3, 0.5),
    (0.5, 1.0),
    (0, 1.0),
]

if __name__ == "__main__":
    exp_name = 'mab-17_2016_10_30_17_08_35_0041'
    folder = exp_name.split("_")[0]
    iclr_folder_name = "data/iclr2016_prereview"
    file_path = os.path.join(config.PROJECT_PATH, "data/s3/{folder}/{exp_name}/params.pkl".format(
        folder=folder,
        exp_name=exp_name

    ))

    tf.reset_default_graph()

    mkdir_p(os.path.join(config.PROJECT_PATH, iclr_folder_name))
    strategy_name = "rnn"
    csv_file = os.path.join(config.PROJECT_PATH, iclr_folder_name, strategy_name + ".csv")

    with open(csv_file, "w") as f:
        writer = csv.DictWriter(
            f,
            ["strategy", "n_arms", "n_episodes", "avg", "stdev", "epsilon_from", "epsilon_to", "best_arm_percent"]
        )
        writer.writeheader()

        with tf.Session() as sess:
            data = joblib.load(file_path)
            policy = data['policy']

            for epsilon in EPSILONS:
                env = MultiEnv(
                    wrapped_env=EpsilonMABEnv(n_arms=5, epsilon=epsilon),
                    n_episodes=500,
                    episode_horizon=1,
                    discount=1
                )

                n_trials = 1000
                n_episodes = 500
                paths = evaluate(env, policy, n_trials, n_episodes)

                n_best_arms = []

                arm_diffs = []

                for path in paths:
                    path_best_arm = np.bincount(env.action_space.unflatten_n(path["actions"])).argmax()
                    true_best_arm = path["env_infos"]["arm_means"][0].argmax()
                    arm_means = np.sort(path["env_infos"]["arm_means"][0].flatten())
                    arm_diff = arm_means[-1] - arm_means[-2]
                    arm_diffs.append(arm_diff)
                    n_best_arms.append(int(path_best_arm == true_best_arm))

                returns = [np.sum(p["rewards"]) for p in paths]

                mean = np.mean(returns)
                std = np.std(returns) / np.sqrt(len(returns) - 1)

                print("Average return:", mean, flush=True)
                print("Std return:", std, flush=True)
                print("Epsilon range:", epsilon, flush=True)
                print("%Best arm:", np.mean(n_best_arms), flush=True)
                print("Avg arm diff", np.mean(arm_diffs), flush=True)
                writer.writerow(dict(
                    strategy=strategy_name,
                    n_arms=5,
                    n_episodes=n_episodes,
                    avg=mena,
                    stdev=std,
                    epsilon_from=epsilon[0],
                    epsilon_to=epsilon[1],
                    best_arm_percent=np.mean(n_best_arms),
                ))

