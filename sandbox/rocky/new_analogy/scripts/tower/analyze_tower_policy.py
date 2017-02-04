import pickle

import matplotlib.pyplot as plt
import numpy as np

from gpr_package.bin import tower_copter_policy as tower
from rllab.sampler.utils import rollout
from sandbox.rocky.new_analogy.scripts.tower.crippled_policy import CrippledPolicy
from sandbox.rocky.s3.resource_manager import resource_manager


def gen_data():
    task_id = tower.get_task_from_text("ab")
    expr = tower.Experiment(2, 1000)
    env = expr.make(task_id)

    all_actions = []

    for idx in range(100):

        print(idx)
        actions = []
        policy = tower.CopterPolicy(task_id)
        obs = env.reset()
        t = 0
        while t < 1000:  # True:
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            # print(reward, done, info, t)
            t += 1
            actions.append(action)
        all_actions.append(actions)

        resource_manager.register_data("tower_copter_actions", np.asarray(all_actions).tobytes())


def analyze_data():
    file_name = resource_manager.get_file("tower_copter_paths_ab")
    with open(file_name, "rb") as f:
        paths = pickle.load(f)
    all_obs = np.concatenate([p["observations"] for p in paths])
    all_actions = np.concatenate([p["actions"] for p in paths])

    print("Mean action:", all_actions.mean(axis=0))
    print("Std action:", all_actions.std(axis=0))
    print("Min action:", all_actions.min(axis=0))
    print("Max action:", all_actions.max(axis=0))
    all_actions -= all_obs[:, :8]
    print("offset Mean action:", all_actions.mean(axis=0))
    print("offset Std action:", all_actions.std(axis=0))
    print("offset Min action:", all_actions.min(axis=0))
    print("offset Max action:", all_actions.max(axis=0))

    # all_actions = np.fromfile(open(file_name, 'rb')).reshape((100, 1000, 8))

    # for idx in range(8):
    #     plt.subplot(4, 4, idx + 1)
    #     plt.hist(all_actions[:, idx], 100)
    #
    # all_actions -= all_obs[:, :8]
    # for idx in range(8):
    #     plt.subplot(4, 4, idx + 1 + 8)
    #     plt.hist(all_actions[:, idx], 100)
    # plt.show()


def cluster_data():
    file_name = resource_manager.get_file("tower_copter_actions")
    all_actions = np.fromfile(open(file_name, 'rb')).reshape((100, 1000, 8))

    import ipdb;
    ipdb.set_trace()


class ActionQuantizer(object):
    def __init__(self, observations, actions, n_bins=100):
        self.observations = np.asarray(observations)
        self.actions = np.asarray(actions)

        self.action_dim = action_dim = actions.shape[-1]
        diff_actions = actions - observations[:, :action_dim]

        action_quants = []

        for action_idx in range(action_dim):
            quants = []
            ith_actions = np.sort(diff_actions[:, action_idx])

            for quant_idx in range(n_bins):
                median = ith_actions[int((quant_idx + 0.5) / n_bins * len(actions))]
                quants.append(median)

            action_quants.append(quants)

        self.action_quants = np.asarray(action_quants)

    def quantize(self, observations, actions):
        observations = np.asarray(observations)
        actions = np.asarray(actions)
        diff_actions = actions - observations[:, :self.action_dim]
        return np.argmin(np.abs(diff_actions[:, :, np.newaxis] - self.action_quants[np.newaxis, :, :]), axis=2)

    def unquantize(self, observations, quantized_actions):
        observations = np.asarray(observations)
        quantized_actions = np.asarray(quantized_actions)
        actions = self.action_quants[
            np.repeat(np.arange(self.action_dim), len(observations)),
            quantized_actions.flatten()
        ].reshape((len(observations), -1))
        return actions + observations[:, :self.action_dim]


def eval_quantized_policy():
    import numpy as np
    # See if a quantized policy can still solve the task
    task_id = tower.get_task_from_text("ab")
    expr = tower.Experiment(2, 1000)
    env = expr.make(task_id)

    file_name = resource_manager.get_file("tower_copter_paths_ab")
    with open(file_name, "rb") as f:
        paths = pickle.load(f)
    all_obs = np.concatenate([p["observations"] for p in paths])
    all_actions = np.concatenate([p["actions"] for p in paths])

    quantizer = ActionQuantizer(all_obs, all_actions, n_bins=50)

    # all_actions = all_actions - all_obs[:, :8]
    #
    # # casting the problem as a 50-way classification...
    # n_bins = 50
    #
    # action_quants = []
    #
    # for action_idx in range(8):
    #     quants = []
    #     ith_actions = np.sort(all_actions[:, action_idx])
    #
    #     for quant_idx in range(n_bins):
    #         median = ith_actions[int((quant_idx + 0.5) / n_bins * len(all_actions))]
    #         quants.append(median)
    #
    #     action_quants.append(quants)
    #
    #     # len(ith_actions)
    #     # # sortenp.sort(ith_actions)
    #     # # np.percentile()
    #     # cnts, boundaries = np.histogram(ith_actions, n_bins)
    #
    # action_quants = np.asarray(action_quants)


    # import ipdb; ipdb.set_trace()


    # all_actions = []

    for idx in range(100):
        print(idx)
        # actions = []
        policy = tower.CopterPolicy(task_id)
        obs = env.reset()
        t = 0
        while t < 1000:  # True:
            action = policy.get_action(obs)
            action = quantizer.unquantize(
                [env.observation_space.flatten(obs)],
                quantizer.quantize([env.observation_space.flatten(obs)], [action])
            )[0]
            # import ipdb; ipdb.set_trace()
            # diff = action - obs[0]#obs[:, :8]
            # # quantize the difference
            # action = obs[0] + action_quants[np.arange(8), np.argmin(np.abs(diff[:, np.newaxis] - action_quants),
            #                                                         axis=1)]


            # quantize this part


            # if action[-1] > 0.4:
            #     action[-1] = 0.5
            # else:
            #     action[-1] = 0.
            obs, reward, done, info = env.step(action)
            # print(reward, done, info, t)
            t += 1
            # actions.append(action)
        # all_actions.append(actions)
        print(reward)

        # resource_manager.register_data("tower_copter_actions", np.asarray(all_actions).tobytes())


def plot_demonstration_diff():
    file_name = resource_manager.get_file("tower_copter_paths_ab_100")
    with open(file_name, "rb") as f:
        paths = pickle.load(f)
    obs = paths[0]['observations']
    actions = paths[0]['actions']
    obs_diff = np.linalg.norm(obs[1:] - obs[:-1], axis=1)
    act_diff = np.linalg.norm(actions[1:] - actions[:-1], axis=1)
    plt.plot(obs_diff, 'r')
    plt.plot(act_diff, 'b')
    plt.plot(np.clip(act_diff / (obs_diff + 1e-8), -10, 10), 'g')
    plt.show()
    # import ipdb; ipdb.set_trace()
    # plt.sub


def run_crippled_policy():
    task_id = tower.get_task_from_text("ab")
    expr = tower.Experiment(2, 2500)
    env = expr.make(task_id)

    policy = CrippledPolicy(tower.CopterPolicy(task_id))
    path = rollout(env, policy, max_path_length=1000)

    obs = path['observations']
    actions = path['actions']
    obs_diff = np.linalg.norm(obs[1:] - obs[:-1], axis=1)
    act_diff = np.linalg.norm(actions[1:] - actions[:-1], axis=1)
    plt.plot(obs_diff, 'r')
    plt.plot(act_diff, 'b')
    plt.plot(np.clip(act_diff / (obs_diff + 1e-8), -10, 10), 'g')
    plt.show()

    # tower.PolicyRunner(env, policy).run()


def fit_crippled_policy():
    """
    Try to run supervised learning on the trajectory generated by the crippled policy
    """
    import tensorflow as tf
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.rocky.new_analogy.tf.algos import Trainer

    task_id = tower.get_task_from_text("ab")
    expr = tower.Experiment(2, 2500)
    env = expr.make(task_id)

    env = GprEnv("tower", gpr_env=env)

    policy = CrippledPolicy(tower.CopterPolicy(task_id))
    path = rollout(env, policy, max_path_length=1000)


    with tf.Session() as sess:
        env = TfEnv(GprEnv(
            "tower",
            task_id=task_id,
            experiment_args=dict(nboxes=2, horizon=1000),
            xinits=[path["env_infos"]["x"][0]],
        ))
        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(256, 256, 256),
            hidden_nonlinearity=tf.nn.tanh,
            name="policy"
        )

        algo = Trainer(
            env=env,
            policy=policy,
            paths=[path],
            n_epochs=1000,
            evaluate_performance=True,
            train_ratio=1.,
            max_path_length=1000,
            n_eval_trajs=1,
            eval_batch_size=10000,
            n_eval_envs=1,
            n_passes_per_epoch=1000,
        )

        algo.train(sess=sess)


if __name__ == "__main__":
    analyze_data()
    # eval_quantized_policy()
    # cluster_data()
    # plot_demonstration_diff()
    # run_crippled_policy()

    # fit_crippled_policy()

#
