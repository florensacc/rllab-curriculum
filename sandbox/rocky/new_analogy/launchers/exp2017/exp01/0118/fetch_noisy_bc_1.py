import pickle

from rllab.misc import logger
from rllab.misc.instrument import run_experiment_lite, variant, VariantGenerator
from sandbox.rocky.tf.envs.base import TfEnv
from sandbox.rocky.tf.spaces import Box

"""
Behavior cloning on fetch using noisy trajectories
"""

MODE = "local_docker"
N_PARALLEL = 1


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11]


def run_task(vv):
    from gpr_package.bin import tower_fetch_policy as tower
    from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv
    import tensorflow as tf
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.new_analogy.tf.algos import Trainer
    from sandbox.rocky.s3.resource_manager import resource_manager
    from rllab.envs.env_spec import EnvSpec
    from sandbox.rocky.new_analogy.fetch_utils import FetchWrapperPolicy

    logger.log("Loading data")
    file_name = resource_manager.get_file("tower_fetch_ab_annotated")
    with open(file_name, "rb") as f:
        paths = pickle.load(f)

    for p in paths:
        p["actions"] = p["taught_actions"]
        del p["taught_actions"]
    logger.log("Loaded")

    task_id = tower.get_task_from_text("ab")
    with tf.Session() as sess:
        env = TfEnv(
            GprEnv(
                "fetch.sim_fetch",
                task_id=task_id,
                experiment_args=dict(nboxes=2, horizon=1000, mocap=True, obs_type="full_state"),
            )
        )

        policy = FetchWrapperPolicy(
            env_spec=env.spec,
            wrapped_policy=GaussianMLPPolicy(
                env_spec=EnvSpec(
                    observation_space=env.observation_space,
                    action_space=Box(low=-10, high=10, shape=(4,))
                ),
                hidden_sizes=(256, 256, 256),
                hidden_nonlinearity=tf.nn.tanh,
                name="policy"
            )
        )

        algo = Trainer(
            env=env,
            policy=policy,
            paths=paths,
            n_epochs=5000,
            n_passes_per_epoch=1,
            evaluate_performance=True,
            train_ratio=0.95,
            max_path_length=1000,
            n_eval_trajs=10,
            eval_batch_size=10000,
            n_eval_envs=10,
            batch_size=1024,
            n_slices=10,
            learn_std=False,
        )

        algo.train(sess=sess)


variants = VG().variants()

print("#Experiments:", len(variants))

for v in variants:

    kwargs = dict(
        use_cloudpickle=True,
        exp_prefix="fetch-noisy-bc-1",
        exp_name="fetch-noisy-bc-1",
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
        del kwargs["env"]["PYTHONPATH"]
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
