import random

from rllab.misc.instrument import run_experiment_lite
from rllab import config
from rllab.misc.instrument import VariantGenerator, variant
import numpy as np

from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.neural_learner.envs.mab_env import MABEnv
from sandbox.rocky.neural_learner.envs.multi_env import MultiEnv

"""
MAB with MPI
"""

USE_GPU = False#True#False#True
USE_CIRRASCALE = False#True
MODE = "ec2"
# MODE = "local_docker"
# MODE = launch_cirrascale("pascal")

# N_RNGS = 100#5#20#100

class VG(VariantGenerator):

    @variant
    def seed(self):
        return [11, 21, 31]#, 41, 51]

    @variant
    def docker_image(self):
        return [
            "dementrock/rllab3-vizdoom-gpu-cuda80:cig",
        ]

    @variant
    def clip_lr(self):
        return [0.1]

    @variant
    def n_arms(self):
        return [10]#, 50]

    @variant
    def n_episodes(self):
        return [500]

    @variant
    def batch_size(self):
        return [500]

    @variant
    def max_path_length(self, n_episodes):
        yield n_episodes

    @variant
    def discount(self):
        return [0.99]

    @variant
    def gae_lambda(self):
        return [0.99, 0.7]

    @variant
    def hidden_dim(self):
        return [128]


vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for vv in variants:

    def run_task(v):
        from sandbox.rocky.neural_learner.algos.parallel_pposgd import RNNActorCritic
        from sandbox.rocky.tf.envs.base import TfEnv
        from sandbox.rocky.neural_learner.algos.parallel_pposgd import ParallelPPOSGD

        env = TfEnv(MultiEnv(wrapped_env=MABEnv(n_arms=v["n_arms"]), n_episodes=v["n_episodes"], episode_horizon=1,
                             discount=v["discount"]))

        ac = RNNActorCritic(name="ac", env_spec=env.spec, hidden_dim=v["hidden_dim"])

        if MODE == "local_docker":
            n_parallel = 32
        else:
            n_parallel = 36

        algo = ParallelPPOSGD(
            env=env,
            ac=ac,
            batch_size=v["batch_size"],
            max_path_length=v["max_path_length"],
            epoch_length=50000,
            discount=v["discount"],
            gae_lambda=v["gae_lambda"],
            opt_epochs=5,
            clip_lr=v["clip_lr"],
            n_parallel=n_parallel
        )

        algo.train()


    config.DOCKER_IMAGE = vv["docker_image"]  # "dementrock/rllab3-vizdoom-gpu-cuda80"
    config.KUBE_DEFAULT_NODE_SELECTOR = {
        "aws/type": "c4.8xlarge",
    }
    config.KUBE_DEFAULT_RESOURCES = {
        "requests": {
            "cpu": 36 * 0.75,
            "memory": "50Gi",
        },
    }
    config.AWS_INSTANCE_TYPE = "c4.8xlarge"
    config.AWS_SPOT = True
    config.AWS_SPOT_PRICE = '1.0'
    config.AWS_REGION_NAME = random.choice(['us-west-2', 'us-west-1', 'us-east-1'])
    config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[config.AWS_REGION_NAME]
    config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[config.AWS_REGION_NAME]
    config.AWS_SECURITY_GROUP_IDS = config.ALL_REGION_AWS_SECURITY_GROUP_IDS[config.AWS_REGION_NAME]

    if MODE == "local_docker":
        env = dict(CUDA_VISIBLE_DEVICES="3")
    else:
        env = dict()

    env['MKL_NUM_THREADS'] = '1'
    env['NUMEXPR_NUM_THREADS'] = '1'
    env['OMP_NUM_THREADS'] = '1'

    run_experiment_lite(
        run_task,
        exp_prefix="mpi-mab-1-1",
        mode=MODE,
        n_parallel=0,
        seed=vv["seed"],
        use_gpu=USE_GPU,
        use_cloudpickle=True,
        variant=vv,
        snapshot_mode="last",
        env=env,
        terminate_machine=True,
        sync_all_data_node_to_s3=False,
    )
    # sys.exit()
