from collections import OrderedDict

from rllab.misc.instrument import run_experiment_lite, VariantGenerator, variant
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.s3.resource_manager import tmp_file_name, resource_manager
from sandbox.rocky.tf.algos.trpo import TRPO
from sandbox.rocky.tf.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv
import tensorflow as tf
import cloudpickle

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

MODE = launch_cirrascale("pascal")


class VG(VariantGenerator):
    @variant
    def task_name(self):
        return TASKS.keys()


def run_task(vv):
    env = TfEnv(normalize(TASKS[vv["task_name"]]()))

    policy = GaussianMLPPolicy(name="policy", env_spec=env.spec)

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        batch_size=50000,
        max_path_length=500,
        n_itr=500,
        env=env,
        policy=policy,
        baseline=baseline,
        gae_lambda=0.99,
    )

    with tf.Session() as sess:
        algo.train(sess)
        file_name = tmp_file_name(file_ext="pkl")
        with open(file_name, "wb") as f:
            cloudpickle.dump(dict(env=env, policy=policy, baseline=baseline), f, protocol=3)
        resource_name = "pretrained_models/{task_name}.pkl".format(task_name=vv["task_name"])
        resource_manager.register_file(resource_name, file_name=file_name)


variants = VG().variants()

print("#Experiments:", len(variants))

for v in variants:
    run_experiment_lite(
        run_task,
        use_cloudpickle=True,
        exp_prefix="trpo-all-1",
        # mode="local",
        variant=v,
        mode=MODE,
        use_gpu=True,
        snapshot_mode="last",
        sync_all_data_node_to_s3=False,
        n_parallel=0,
        docker_image="dementrock/rllab3-shared-gpu-cuda80",
    )
