# First experiment with claw, using behavior cloning
# The end goal is to get some reasonable behavior using as few demonstrations as possible
from rllab.misc.instrument import run_experiment_lite, variant, VariantGenerator
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.new_analogy.policies.normalizing_policy import NormalizingPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc import logger
import numpy as np

# MODE = "local_docker"
MODE = launch_cirrascale("pascal")
N_PARALLEL = 8


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]

    @variant
    def init_std(self):
        return [0.3]

    @variant
    def n_trajs(self):
        return ["500"]

    @variant
    def bc_n_epochs(self):
        return [100, 500, 1000]


def run_task(vv):
    from sandbox.rocky.tf.algos.trpo import TRPO
    from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv
    import tensorflow as tf
    from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy

    from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
    from sandbox.rocky.new_analogy.algos.ff_bc_trainer import Trainer

    with tf.Session() as sess:
        logger.log("Loading data...")
        data = np.load("/shared-data/claw-{n_trajs}-data.npz".format(n_trajs=vv["n_trajs"]))
        exp_x = data["exp_x"]
        exp_u = data["exp_u"]
        exp_rewards = data["exp_rewards"]
        success_ids = np.where(exp_rewards[:, -1] >= 4.5)[0]
        exp_x = exp_x[success_ids]
        exp_u = exp_u[success_ids]
        exp_rewards = exp_rewards[success_ids]

        logger.log("Loaded")

        paths = []

        for xs, us, rewards in zip(exp_x, exp_u, exp_rewards):
            paths.append(dict(observations=xs, actions=us, rewards=rewards))

        env = TfEnv(GprEnv("TF2", xinits=exp_x[:, 0, :]))

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(256, 256, 256),
            hidden_nonlinearity=tf.nn.relu,
            name="policy"
        )

        bc_algo = Trainer(
            env=env,
            policy=policy,
            paths=paths,
            n_epochs=vv["bc_n_epochs"],
            evaluate_performance=False,
            train_ratio=0.9,
        )

        bc_algo.train(sess=sess)

        baseline = GaussianMLPBaseline(
            env_spec=env.spec,
            regressor_args=dict(
                use_trust_region=True,
                hidden_sizes=(128, 128),
                optimizer=ConjugateGradientOptimizer(),
                step_size=0.1,
            ),
        )

        sess.run(tf.assign(policy._l_std_param.param, [np.log(vv["init_std"])] * env.action_dim))

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=50000,
            max_path_length=100,
            n_itr=5000,
            discount=0.995,
            gae_lambda=0.97,
            parallel_vec_env=True,
            n_vectorized_envs=100,
        )

        algo.train(sess=sess)


variants = VG().variants()

print("#Experiments:", len(variants))

for v in variants:
    run_experiment_lite(
        run_task,
        use_cloudpickle=True,
        exp_prefix="trpo-joint-bc-claw-2",
        mode=MODE,
        use_gpu=True,
        snapshot_mode="last",
        sync_all_data_node_to_s3=False,
        n_parallel=N_PARALLEL,
        env=dict(PYTHONPATH="/root/code/rllab:/root/code/rllab/conopt_root"),
        docker_image="quay.io/openai/rocky-rllab3-conopt-gpu-pascal",
        docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
        variant=v,
        seed=v["seed"],
    )
