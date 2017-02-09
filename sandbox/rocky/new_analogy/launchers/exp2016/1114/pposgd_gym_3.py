import random

import cloudpickle
import tensorflow as tf

from rllab import config
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite, VariantGenerator, variant
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.new_analogy.tf.algos.pposgd_joint_ac import PPOSGD
from sandbox.rocky.s3.resource_manager import tmp_file_name, resource_manager
from sandbox.rocky.tf.envs.base import TfEnv


MODE = "local_docker"
MODE = "local"
MODE = "ec2"
MODE = launch_cirrascale("pascal")

USE_GPU = True

if MODE in ["local_docker", "local", "ec2"]:
    USE_GPU = False


class VG(VariantGenerator):
    @variant
    def task_name(self):
        return [
            "Hopper-v1",
            # "Humanoid-v1",
            # "HalfCheetah-v1"
        ]

    @variant
    def seed(self):
        return [1, 2, 3]

    @variant
    def opt_n_epochs(self):
        return [2, 5, 10]

    @variant
    def opt_batch_size(self):
        return [64, 128, 256]


def run_task(vv):
    env = TfEnv(normalize(GymEnv(
        vv["task_name"],
        record_log=False,
        record_video=False,
    )))

    max_path_length = env.wrapped_env.wrapped_env.env.spec.timestep_limit

    from sandbox.rocky.new_analogy.tf.policies.gaussian_mlp_actor_critic import GaussianMLPActorCritic

    ac = GaussianMLPActorCritic(
        name="ac",
        env_spec=env.spec,
        hidden_sizes=(128, 128),
        hidden_nonlinearity=tf.nn.relu,
    )

    from sandbox.rocky.neural_learner.optimizers.sgd_optimizer import SGDOptimizer
    algo = PPOSGD(
        batch_size=50000,
        max_path_length=max_path_length,
        n_itr=500,
        env=env,
        policy=ac,
        baseline=ac,
        discount=0.995,
        gae_lambda=0.97,
        min_n_epochs=vv["opt_n_epochs"],
        parallel_vec_env=True,
        optimizer=SGDOptimizer(
            n_epochs=vv["opt_n_epochs"],
            batch_size=vv["opt_batch_size"],
        ),
    )

    with tf.Session() as sess:
        algo.train(sess)
        file_name = tmp_file_name(file_ext="pkl")
        with open(file_name, "wb") as f:
            cloudpickle.dump(dict(env=env, policy=ac, baseline=ac), f, protocol=3)
        resource_name = "pretrained_models/pposgd_{task_name}_seed_{seed}.pkl".format(
            task_name=vv["task_name"],
            seed=vv["seed"]
        )
        resource_manager.register_file(resource_name, file_name=file_name)


variants = VG().variants()

print("#Experiments:", len(variants))

for v in variants:

    config.AWS_INSTANCE_TYPE = "c4.2xlarge"
    config.AWS_SPOT = True
    config.AWS_SPOT_PRICE = '1.0'
    config.AWS_REGION_NAME = random.choice(
        ['us-west-2', 'us-west-1', 'us-east-1', 'ap-southeast-2', 'ap-south-1', 'ap-northeast-1']
    )
    config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[config.AWS_REGION_NAME]
    config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[config.AWS_REGION_NAME]
    config.AWS_SECURITY_GROUP_IDS = config.ALL_REGION_AWS_SECURITY_GROUP_IDS[config.AWS_REGION_NAME]

    run_experiment_lite(
        run_task,
        use_cloudpickle=True,
        exp_prefix="pposgd-gym-3",
        variant=v,
        mode=MODE,
        use_gpu=USE_GPU,
        snapshot_mode="last",
        sync_all_data_node_to_s3=False,
        n_parallel=4,
        docker_image="dementrock/rllab3-shared-gpu-cuda80",
        seed=v["seed"]
    )
    # break
