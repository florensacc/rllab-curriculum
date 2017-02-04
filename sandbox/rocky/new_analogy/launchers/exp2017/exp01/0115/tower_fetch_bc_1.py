from rllab.misc import logger
from rllab.misc.instrument import run_experiment_lite, variant, VariantGenerator
from rllab.sampler import parallel_sampler
from sandbox.rocky.new_analogy.scripts.tower.gpr_policy_wrapper import GprPolicyWrapper
from sandbox.rocky.tf.envs.base import TfEnv

"""
Only behavior clone the last few time steps
"""

MODE = "local_docker"  # _docker"  # _docker"
# MODE = launch_cirrascale("pascal")
N_PARALLEL = 32  # 1#8


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11]  # , 21, 31, 41, 51]


def run_task(vv):
    from gpr_package.bin import tower_copter_policy as tower
    from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv
    import tensorflow as tf
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.new_analogy.tf.policies.randomize_policy_wrapper import RandomizePolicyWrapper
    from sandbox.rocky.new_analogy.tf.algos import Trainer
    import numpy as np


    # resource_manager.get_file(
    #     resource_name="fetch_v1",
    #     mkfile=gen_data
    # )

    def gen_data():
        from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv
        from gpr_package.bin import tower_fetch_policy as tower
        np.random.seed(0)
        task_id = tower.get_task_from_text("ab")

        horizon = 1000

        means = np.array(
            [0.26226245, 0.04312864, 0.61919527, 0., 1.57,
             0., 0.13145038, 0.13145038]
        )
        stds = np.array([9.87383461e-02, 1.34582481e-01, 5.18472679e-02,
                         0.00000000e+00, 4.67359484e-12, 0.00000000e+00,
                         9.91322751e-01, 9.91322751e-01])

        # means, stds = compute_stds(gpr_env, xinit)

        env = TfEnv(GprEnv("fetch.sim_fetch", task_id=task_id, experiment_args=dict(nboxes=2, horizon=horizon)))
        policy = RandomizePolicyWrapper(
            GprPolicyWrapper(tower.FetchPolicy(task_id)),
            action_mean=means,
            action_std=stds,
            noise_scale=0.1,
            p_random_action=0.,#0.1,
        )
        parallel_sampler.populate_task(env=env, policy=policy)

        n_trajs = 100
        paths = parallel_sampler.sample_paths(None, max_samples=n_trajs * horizon,
                                              max_path_length=horizon)
        print("Success rate: ", np.mean(np.asarray([p["rewards"][-1] for p in paths]) > 4))
        return paths

    with tf.Session() as sess:
        logger.log("Loading data...")
        paths = gen_data()

        logger.log("Loaded")

        task_id = tower.get_task_from_text("ab")

        env = TfEnv(GprEnv("fetch.sim_fetch", task_id=task_id, experiment_args=dict(nboxes=2, horizon=1000)))

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(1024, 1024),#256, 256),
            hidden_nonlinearity=tf.nn.tanh,
            name="policy"
        )

        algo = Trainer(
            env=env,
            policy=policy,
            paths=paths,
            n_epochs=1000,
            evaluate_performance=True,
            train_ratio=0.9,
            max_path_length=1000,
            n_eval_trajs=5,
            eval_batch_size=5000,
            n_eval_envs=5,
            n_passes_per_epoch=10,
        )

        algo.train(sess=sess)


variants = VG().variants()

print("#Experiments:", len(variants))

for v in variants:

    kwargs = dict(
        use_cloudpickle=True,
        exp_prefix="tower-bc-1",
        mode=MODE,
        use_gpu=True,
        snapshot_mode="last",
        sync_all_data_node_to_s3=False,
        n_parallel=N_PARALLEL,
        env=dict(CUDA_VISIBLE_DEVICES="1", PYTHONPATH="/root/code/rllab:/root/code/rllab/gpr_package"),
        variant=v,
        seed=v["seed"],
    )

    if MODE == "local":
        del kwargs["env"]["PYTHONPATH"]  # =
    else:
        kwargs = dict(
            kwargs,
            docker_image="quay.io/openai/rocky-rllab3-gpr-gpu-pascal:20170114",
            docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
        )

    run_experiment_lite(
        run_task,
        **kwargs
    )
