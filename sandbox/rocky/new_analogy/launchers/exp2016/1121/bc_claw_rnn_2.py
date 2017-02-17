# First experiment with claw, using behavior cloning
# The end goal is to get some reasonable behavior using as few demonstrations as possible
import numpy as np
import tensorflow as tf

from rllab.misc import logger
from rllab.misc.instrument import run_experiment_lite, variant, VariantGenerator
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.tf.envs.base import TfEnv

"""
A wider range of architectural search
"""

MODE = launch_cirrascale("pascal")
# MODE = "local_docker"


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]

    @variant
    def n_trajs(self):
        return ["20k"]

    @variant
    def learning_rate(self):
        return [1e-3]  # , 1e-4]

    @variant
    def opt_batch_size(self):
        return [128]

    @variant
    def opt_n_steps(self):
        return [None]  # , 20]

    @variant(hide=True)
    def rnn_cell(self):
        return list(dict(
            gru=tf.nn.rnn_cell.GRUCell(num_units=256),
            # basic_lstm=tf.nn.rnn_cell.BasicLSTMCell(num_units=256, state_is_tuple=False),
            # basic_gated=rnn_cell_modern.BasicGatedCell(num_units=256),
            # mgu=rnn_cell_modern.MGUCell(num_units=256),
            # gru_layer_norm=rnn_cell_layernorm_modern.GRUCell_LayerNorm(num_units=256),
            # basic_lstm_layer_norm=rnn_cell_layernorm_modern.BasicLSTMCell_LayerNorm(num_units=256),
            # gru_mulint_layer_norm=rnn_cell_mulint_layernorm_modern.GRUCell_MulInt_LayerNorm(num_units=256),
            # basic_lstm_mulint_layer_norm=rnn_cell_mulint_layernorm_modern.BasicLSTMCell_MulInt_LayerNorm(num_units=256),
        ).items())

    @variant
    def rnn_cell_type(self, rnn_cell):
        yield rnn_cell[0]

    @variant
    def n_layers(self):
        return [1]#3, 1]


def run_task(vv):
    logger.log("Loading data...")
    data = np.load("/shared-data/claw-{n_trajs}-data.npz".format(n_trajs=vv["n_trajs"]))
    exp_x = data["exp_x"]
    exp_u = data["exp_u"]
    exp_rewards = data["exp_rewards"]
    logger.log("Loaded")

    from sandbox.rocky.new_analogy.tf.algos.rnn_bc_trainer import Trainer
    # from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.tf.policies.gaussian_tf_rnn_policy import GaussianTfRNNPolicy
    from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv

    env = TfEnv(GprEnv("TF2"))

    if vv["n_layers"] == 1:
        cell = vv["rnn_cell"][1]
    else:
        cell = tf.nn.rnn_cell.MultiRNNCell([vv["rnn_cell"][1]] * vv["n_layers"], state_is_tuple=False)

    policy = GaussianTfRNNPolicy(
        env_spec=env.spec,
        cell=cell,
        state_include_action=False,
        name="policy",
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
        evaluate_performance=MODE not in ["local_docker"],
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
        exp_prefix="bc-rnn-claw-2",
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
