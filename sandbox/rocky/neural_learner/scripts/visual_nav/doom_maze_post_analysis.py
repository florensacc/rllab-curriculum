import tempfile

from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.s3.resource_manager import resource_manager
import joblib
import tensorflow as tf
from rllab.misc.instrument import run_experiment_lite, VariantGenerator, variant
import numpy as np
from rllab.misc import logger

MODE = "local"
USE_GPU = True
MODE = launch_cirrascale("pascal")




class VG(VariantGenerator):
    @variant
    def seed(self):
        return [1]

    @variant
    def n_trajs(self):
        return [1000]

    @variant
    def n_episodes(self):
        return [5]#2, 3, 4, 5]

    @variant
    def episode_horizon(self, maze_size):
        if maze_size == 3:
            return [250]
        return [1000]#250, 500]

    @variant
    def maze_size(self):
        return [3, 5]#3, 5]


vg = VG()
variants = vg.variants()

for v in variants:

    kwargs = [
        ("n_trajs", v["n_trajs"]),
        ("n_episodes", v["n_episodes"]),
        ("episode_horizon", v["episode_horizon"]),
        ("maze_size", v["maze_size"]),
        ("seed", v["seed"]),
        ("version", "v5"),
    ]

    resource_name = "_".join("{0}_{1}".format(k, v) for k, v in kwargs)

    resource_name = "doom-maze-analysis/%s.npz" % resource_name

    try:
        file_name = resource_manager.get_file(resource_name)
    except:
        continue

    data = np.load(file_name)

    traj_lens = data['traj_lens']
    traj_success = data['traj_success']

    n_episodes = traj_lens.shape[1]

    print("Maze size: %d" % v["maze_size"])
    print("#episodes: %d" % v["n_episodes"])
    print("episode horizon: %d" % v["episode_horizon"])

    success_rates = []
    mean_traj_lens = []
    std_lens = []

    # mask = np.cast['bool'](traj_success[:, 0])
    diff_trajlens = traj_lens[:, 0] - traj_lens[:, 1]
    improved = diff_trajlens >= 0#- 10#traj_lens[:, 0] * 0.2#1

    print("%%improved: %.2f%%" % (np.mean(improved)*100))

    for idx in range(n_episodes):
        success_rate = np.mean(traj_success[:, idx])
        mask = np.cast['bool'](traj_success[:, idx])
        mean_traj_len = np.mean(traj_lens[mask, idx])
        std = np.std(traj_lens[mask, idx]) / np.sqrt(len(traj_lens))
        print("%dth success rate: %.1f%%" % (idx, success_rate * 100))
        print("%dth mean success traj len : %.1f \\pm %.1f" % (idx, mean_traj_len, std))
        success_rates.append(success_rate)
        mean_traj_lens.append(mean_traj_len)
        std_lens.append(std)

    print("\n".join("$%.1f\\%%$" % (x*100) for x in success_rates))

    print("\n".join("$%.1f \\pm %.1f$" % (x, y) for x,y in zip(mean_traj_lens, std_lens)))

    print()

    # import ipdb; ipdb.set_trace()

#     run_experiment_lite(
#         run_task,
#         exp_prefix="doom-maze-analysis-1",
#         mode=MODE,
#         n_parallel=0,
#         seed=vv["seed"],
#         use_gpu=USE_GPU,
#         use_cloudpickle=True,
#         variant=vv,
#         docker_image="dementrock/rllab3-vizdoom-gpu-cuda80:cig",
#         # snapshot_mode="last",
#         # env=env,
#         terminate_machine=True,
#         sync_all_data_node_to_s3=False,
#     )
#
#     # sys.exit()
# # run_task(dict(n_trajs=4, n_episodes=2, episode_horizon=250, maze_size=3, seed=1))
#
#
# # run_experiment_lite(run_task, mode="local_docker", docker_image="dementrock/rllab3-vizdoom-gpu-cuda80:cig",
# #                     use_cloudpickle=True)