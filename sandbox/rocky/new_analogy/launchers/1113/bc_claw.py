# First experiment with claw, using behavior cloning
# The end goal is to get some reasonable behavior using as few demonstrations as possible
from rllab.misc.instrument import run_experiment_lite, variant, VariantGenerator
import pickle
import numpy as np
from rllab.misc import logger
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.tf.envs.base import TfEnv


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31, 41, 51]

    @variant
    def n_trajs(self):
        return ["20k", "2k", "500"]

    @variant
    def learning_rate(self):
        return [1e-3, 1e-4]


def run_task(vv):
    logger.log("Loading data...")
    data = np.load("/shared-data/claw-{n_trajs}-data.npz".format(n_trajs=vv["n_trajs"]))
    exp_x = data["exp_x"]
    exp_u = data["exp_u"]
    exp_rewards = data["exp_rewards"]
    logger.log("Loaded")

    from sandbox.rocky.new_analogy.algos.ff_bc_trainer import Trainer
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.new_analogy.envs.conopt_env import ConoptEnv
    import tensorflow as tf

    env = TfEnv(ConoptEnv("TF2"))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(256, 256, 256),
        hidden_nonlinearity=tf.nn.relu,
        name="policy"
    )

    paths = []

    for xs, us, rewards in zip(exp_x, exp_u, exp_rewards):
        paths.append(dict(observations=xs, actions=us, rewards=rewards))

    algo = Trainer(
        env=env,
        policy=policy,
        paths=paths,
        threshold=4.5,
        n_epochs=5000,
    )

    algo.train()


variants = VG().variants()

print("#Experiments:", len(variants))

for v in variants:
    run_experiment_lite(
        run_task,
        use_cloudpickle=True,
        exp_prefix="bc-claw-2",
        mode=launch_cirrascale("pascal"),
        use_gpu=True,
        snapshot_mode="last",
        sync_all_data_node_to_s3=False,
        n_parallel=0,
        env=dict(PYTHONPATH="/root/code/rllab:/root/code/rllab/conopt_root"),
        docker_image="quay.io/openai/rocky-rllab3-conopt-gpu-pascal",
        docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
        variant=v,
        seed=v["seed"],
    )
