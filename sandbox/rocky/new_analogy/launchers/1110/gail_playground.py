from collections import OrderedDict

import tensorflow as tf

from rllab.misc.instrument import VariantGenerator, variant, run_experiment_lite
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.new_analogy.discriminators.mlp_discriminator import MLPDiscriminator
from sandbox.rocky.new_analogy.sample_processors.gail_sample_processor import GAILSampleProcessor
from sandbox.rocky.s3.resource_manager import resource_manager
from rllab.envs.normalized_env import normalize
import cloudpickle

from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.envs.parallel_vec_env_executor import ParallelVecEnvExecutor
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.box2d.double_pendulum_env import DoublePendulumEnv
from rllab.envs.box2d.mountain_car_env import MountainCarEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.mujoco.inverted_double_pendulum_env import InvertedDoublePendulumEnv
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.mujoco.walker2d_env import Walker2DEnv
from rllab.envs.mujoco.ant_env import AntEnv

MODE = launch_cirrascale("pascal")

TASKS = OrderedDict([
    ("cartpole", CartpoleEnv),
    ("cartpole_swing_up", CartpoleSwingupEnv),
    ("double_pendulum", DoublePendulumEnv),
    ("mountain_car", MountainCarEnv),
    ("half_cheetah", HalfCheetahEnv),
    ("hopper", HopperEnv),
    ("inverted_double_pendulum", InvertedDoublePendulumEnv),
    ("swimmer", SwimmerEnv),
    ("walker2d", Walker2DEnv),
    ("ant", AntEnv)
])


class VG(VariantGenerator):
    @variant
    def task(self):
        return [
            # "cartpole",
            # "cartpole_swing_up",
            "double_pendulum",
            # "mountain_car",
            # "inverted_double_pendulum"
        ]

    @variant
    def disc_n_epochs(self):
        return [2]  # , 5, 10]

    @variant
    def seed(self):
        return [1, 2, 3]


def run_task(vv):
    with tf.Session() as sess:
        # traj_resource_name = "demo_trajs/cartpole_1000.pkl"

        task = vv["task"]
        n_trajs = 1000
        horizon = 500
        batch_size = 50000
        n_envs = batch_size // horizon

        deterministic = False  # True

        traj_resource_name = "demo_trajs/{task}_n_trajs_{n_trajs}_horizon_{horizon}_deterministic_{" \
                             "deterministic}.pkl".format(task=task, n_trajs=str(n_trajs), horizon=str(
            horizon), deterministic=str(deterministic))

        with open(resource_manager.get_file(traj_resource_name), "rb") as f:
            demo_paths = cloudpickle.load(f)

        env = TfEnv(normalize(TASKS[task]()))

        discriminator = MLPDiscriminator(env_spec=env.spec, demo_paths=demo_paths, n_epochs=v["disc_n_epochs"])

        policy = GaussianMLPPolicy(name="policy", env_spec=env.spec)
        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=batch_size,
            max_path_length=horizon,
            entropy_bonus_coeff=1e-3,
            n_itr=500,
            sampler=VectorizedSampler(env=env, policy=policy, n_envs=n_envs, vec_env=ParallelVecEnvExecutor(
                env=env, n_envs=n_envs,
            )),
            sample_processor_cls=GAILSampleProcessor,
            sample_processor_args=dict(discriminator=discriminator),
        )

        algo.train(sess=sess)


variants = VG().variants()

print("#Experiments:", len(variants))

for v in variants:
    run_experiment_lite(
        run_task,
        use_cloudpickle=True,
        exp_prefix="gail-4",
        variant=v,
        mode=MODE,
        use_gpu=True,
        snapshot_mode="last",
        sync_all_data_node_to_s3=False,
        n_parallel=8,
        docker_image="dementrock/rllab3-shared-gpu-cuda80",
        seed=v["seed"],
    )
