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


def run_task(v):

    # resource_name = "saved_params/doom-maze/doom-maze-77_2016_10_30_22_11_44_0004.pkl"

    # resource_name = "saved_params/doom-maze/doom-maze-77_2016_10_30_22_11_44_0012.pkl"

    # resource_name = "saved_params/doom-maze-v2/doom-maze-77_2016_10_30_22_11_44_0007.pkl"
    resource_name = "saved_params/doom-maze-v3/doom-maze-77_2016_10_30_22_11_44_0007.pkl"

    kwargs = [
        ("n_trajs", v["n_trajs"]),
        ("n_episodes", v["n_episodes"]),
        ("episode_horizon", v["episode_horizon"]),
        ("maze_size", v["maze_size"]),
        ("seed", v["seed"]),
        ("version", "v5"),
    ]

    data_resource_name = "_".join("{0}_{1}".format(k, v) for k, v in kwargs)

    data_resource_name = "doom-maze-analysis/%s.npz" % data_resource_name

    file_name = resource_manager.get_file(resource_name)

    with tf.Session() as sess:
        data = joblib.load(file_name)

        from sandbox.rocky.neural_learner.envs.doom_fixed_goal_finding_maze_env import DoomFixedGoalFindingMazeEnv
        from sandbox.rocky.tf.envs.base import TfEnv
        from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv
        env = TfEnv(
            MultiEnv(
                wrapped_env=DoomFixedGoalFindingMazeEnv(
                    rescale_obs=(30, 40),
                    reset_map=False,
                    living_reward=-0.01,
                    frame_skip=4,
                    allow_backwards=False,
                    margin=1,
                    side_length=96,
                    n_trajs=1000,
                    maze_sizes=[v["maze_size"]],
                    seed=v["seed"],
                    version="v1",
                    map_preset=None,
                    wall_penalty=0.001,
                ),
                n_episodes=v["n_episodes"],
                episode_horizon=v["episode_horizon"],
                discount=0.99,
            )
        )

        max_path_length = v["n_episodes"] * v["episode_horizon"]

        from sandbox.rocky.neural_learner.samplers.vectorized_sampler import VectorizedSampler
        sampler = VectorizedSampler(env=env, policy=data['policy'], n_envs=min(v["n_trajs"], 128))
        sampler.start_worker()

        paths = sampler.obtain_samples(0, max_path_length=max_path_length, batch_size=max_path_length * v["n_trajs"],
                                       max_n_trajs=v["n_trajs"])

        # path_lens = [len(p["rewards"]) for p in paths]

        traj_lens_list = []
        traj_success_list = []

        for path in paths:
            rewards = path['rewards']
            splitter = np.where(path['env_infos']['episode_done'])[0][:-1] + 1
            split_rewards = np.split(rewards, splitter)
            successes = [rs[-1] > 0 for rs in split_rewards]
            traj_lens = [len(x) for x in split_rewards]
            # import ipdb; ipdb.set_trace()
            # terms = [o[-1] for o in p["observations"]]
            # end_ids = np.where(np.append(terms, 1))[0][1:]
            # start_ids = np.insert(end_ids[:-1], 0, 0)
            # traj_lens = end_ids - start_ids
            traj_lens_list.append(traj_lens)
            traj_success_list.append(successes)

        traj_lens_list = np.asarray(traj_lens_list)
        traj_success_list = np.asarray(traj_success_list)

        f_name = tempfile.NamedTemporaryFile().name + ".npz"

        np.savez_compressed(f_name, traj_lens=traj_lens_list, traj_success=traj_success_list)

        resource_manager.register_file(data_resource_name, file_name=f_name)

vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for vv in variants:
    run_experiment_lite(
        run_task,
        exp_prefix="doom-maze-analysis-5",
        mode=MODE,
        n_parallel=0,
        seed=vv["seed"],
        use_gpu=USE_GPU,
        use_cloudpickle=True,
        variant=vv,
        docker_image="dementrock/rllab3-vizdoom-gpu-cuda80:cig",
        # snapshot_mode="last",
        # env=env,
        terminate_machine=True,
        sync_all_data_node_to_s3=False,
    )

    # sys.exit()
# run_task(dict(n_trajs=4, n_episodes=2, episode_horizon=250, maze_size=3, seed=1))


# run_experiment_lite(run_task, mode="local_docker", docker_image="dementrock/rllab3-vizdoom-gpu-cuda80:cig",
#                     use_cloudpickle=True)