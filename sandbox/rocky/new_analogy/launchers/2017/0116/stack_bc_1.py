from rllab.misc.instrument import run_experiment_lite, variant, VariantGenerator
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.new_analogy.policies.normalizing_policy import NormalizingPolicy
from sandbox.rocky.new_analogy.policies.residual_policy import ResidualPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc import logger
import pickle

"""
Behavior clone single trajectory
"""

MODE = "local_docker"  # _docker"  # _docker"
# MODE = launch_cirrascale("pascal")
N_PARALLEL = 1#8


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11]#, 21, 31]#, 41, 51]




def run_task(vv):
    from gpr_package.bin import tower_copter_policy as tower
    from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv
    import tensorflow as tf
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.new_analogy.algos.ff_bc_trainer import Trainer
    from sandbox.rocky.s3.resource_manager import resource_manager
    import os

    logger.log("Loading data...")
    paths = []
    root_path = "/shared-data/stack_ab_trajs_v2_processed"
    for traj_file in os.listdir(root_path):
        with open(os.path.join(root_path, traj_file), "rb") as f:
            paths.append(pickle.load(f))

    # import ipdb; ipdb.set_trace()
    #
    # file_name = resource_manager.get_file("tower_copter_paths_ab_crippled_100")
    # with open(file_name, 'rb') as f:
    #     paths = pickle.load(f)
    logger.log("Loaded")

    # from gpr.envs.stack import Experiment
    # expr = Experiment(nboxes=2, horizon=1000)
    # env = expr.make(task_id=task_id)

    # import ipdb; ipdb.set_trace()


    with tf.Session() as sess:

        task_id = [[1, "top", 0]]

        env = TfEnv(
            GprEnv(
                "stack",
                task_id=task_id,
                experiment_args=dict(nboxes=2, horizon=300),
                # xinits=xinits[:1],
            )
        )

        paths = list(filter(env.wrapped_env._is_success, paths))
        print("#paths:", len(paths))
        # xinits = []
        # for path in paths:
        #     xinits.append(path["env_infos"]["x"][0])
        # env = TfEnv(
        #     GprEnv(
        #         "stack",
        #         task_id=task_id,
        #         experiment_args=dict(nboxes=2, horizon=1000),
        #         xinits=xinits[:1],
        #     )
        # )

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(256, 256, 256),
            hidden_nonlinearity=tf.nn.tanh,
            name="policy"
        )#)

        algo = Trainer(
            env=env,
            policy=policy,
            paths=paths,#[:1],
            n_epochs=5000,
            n_passes_per_epoch=10,#100,
            evaluate_performance=True,
            # n_passes_per_epoch=1,
            train_ratio=0.9,#1.,#0.9,
            max_path_length=300,
            n_eval_trajs=50,
            eval_batch_size=15000,
            n_eval_envs=50,
            batch_size=512,
            n_slices=10,
            learn_std=False,#True,
        )

        algo.train(sess=sess)


variants = VG().variants()

print("#Experiments:", len(variants))

for v in variants:

    kwargs = dict(
        use_cloudpickle=True,
        exp_prefix="stack-bc-1",
        exp_name="stack-bc-1",
        mode=MODE,
        use_gpu=True,
        snapshot_mode="last",
        sync_all_data_node_to_s3=False,
        n_parallel=N_PARALLEL,
        env=dict(CUDA_VISIBLE_DEVICES="4", PYTHONPATH="/root/code/rllab:/root/code/rllab/gpr_package"),
        variant=v,
        seed=v["seed"],
    )

    if MODE == "local":
        del kwargs["env"]["PYTHONPATH"]  # =
    else:
        kwargs = dict(
            kwargs,
            docker_image="dementrock/rocky-rllab3-gpr-gpu-pascal:20170115",
            docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
        )

    run_experiment_lite(
        run_task,
        **kwargs
    )
