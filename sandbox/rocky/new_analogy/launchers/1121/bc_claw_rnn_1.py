# First experiment with claw, using behavior cloning
# The end goal is to get some reasonable behavior using as few demonstrations as possible
import numpy as np

from rllab.misc import logger
from rllab.misc.instrument import run_experiment_lite, variant, VariantGenerator
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.tf.envs.base import TfEnv

"""
First test using RNNs
"""

MODE = launch_cirrascale("pascal")
# MODE = "local_docker"


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]#, 41, 51]

    @variant
    def n_trajs(self):
        return ["20k"]  # "500", "20k", "2k"]

    @variant
    def learning_rate(self):
        return [1e-3, 1e-4]

    @variant
    def opt_batch_size(self):
        return [128]

    @variant
    def opt_n_steps(self):
        return [None, 20]

    @variant
    def nonlinearity(self):
        return ["tanh", "relu"]


def run_task(vv):
    logger.log("Loading data...")
    data = np.load("/shared-data/claw-{n_trajs}-data.npz".format(n_trajs=vv["n_trajs"]))
    exp_x = data["exp_x"]
    exp_u = data["exp_u"]
    exp_rewards = data["exp_rewards"]
    logger.log("Loaded")

    from sandbox.rocky.new_analogy.algos.rnn_bc_trainer import Trainer
    # from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.tf.policies.gaussian_rnn_policy import GaussianRNNPolicy
    from sandbox.rocky.tf.policies.rnn_utils import NetworkType
    from sandbox.rocky.new_analogy.envs.conopt_env import ConoptEnv
    import tensorflow as tf

    env = TfEnv(ConoptEnv("TF2"))

    policy = GaussianRNNPolicy(
        env_spec=env.spec,
        hidden_dim=256,
        hidden_nonlinearity=getattr(tf.nn, vv["nonlinearity"]),
        network_type=NetworkType.TF_BASIC_LSTM,
        weight_normalization=True,
        layer_normalization=False,
        state_include_action=False,
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
        opt_batch_size=vv["opt_batch_size"],
        opt_n_steps=vv["opt_n_steps"],
        learning_rate=vv["learning_rate"],
    )

    algo.train()


variants = VG().variants()

print("#Experiments:", len(variants))

env = dict(PYTHONPATH="/root/code/rllab:/root/code/rllab/conopt_root")

if MODE in ["local_docker"]:
    env["CUDA_VISIBLE_DEVICES"] = "1"

for v in variants:
    run_experiment_lite(
        run_task,
        use_cloudpickle=True,
        exp_prefix="bc-rnn-claw-1-1",
        mode=MODE,
        use_gpu=True,
        snapshot_mode="last",
        sync_all_data_node_to_s3=False,
        n_parallel=0,
        env=env,
        docker_image="quay.io/openai/rocky-rllab3-conopt-gpu-pascal",
        docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
        variant=v,
        seed=v["seed"],
    )
