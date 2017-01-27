from rllab.misc.instrument import run_experiment_lite, variant, VariantGenerator
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale
from sandbox.rocky.new_analogy.policies.normalizing_policy import NormalizingPolicy
from sandbox.rocky.new_analogy.policies.residual_policy import ResidualPolicy
from sandbox.rocky.tf.envs.base import TfEnv
from rllab.misc import logger
import pickle

"""
Behavior clone using the crippled policy
"""

MODE = "local_docker"  # _docker"  # _docker"
# MODE = launch_cirrascale("pascal")
N_PARALLEL = 1#8


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31, 41, 51]


def run_task(vv):
    from gpr_package.bin import tower_copter_policy as tower
    from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv
    import tensorflow as tf
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.new_analogy.algos.rnn_bc_trainer import Trainer
    from sandbox.rocky.s3.resource_manager import resource_manager
    from sandbox.rocky.tf.policies.gaussian_rnn_policy import GaussianRNNPolicy

    with tf.Session() as sess:
        logger.log("Loading data...")
        file_name = resource_manager.get_file("tower_copter_paths_ab_crippled_10000")
        with open(file_name, 'rb') as f:
            paths = pickle.load(f)
        logger.log("Loaded")

        # truncate the paths to the last 200 time steps
        # xinits = []
        # for path in paths:
        #     path["observations"] = path["observations"][800:]
        #     path["actions"] = path["actions"][800:]
        #     path["rewards"] = path["rewards"][800:]
        #     path["env_infos"] = {k: v[800:] for k, v in path["env_infos"].items()}
        #     xinits.append(path["env_infos"]["x"][0])

        task_id = tower.get_task_from_text("ab")
        # expr = tower.Experiment(2, 1000)

        env = TfEnv(GprEnv("tower", task_id=task_id, experiment_args=dict(nboxes=2, horizon=1000)))#, xinits=xinits))

        policy = GaussianRNNPolicy(
            env_spec=env.spec,
            name="policy",
            hidden_dim=256,
            hidden_nonlinearity=tf.nn.tanh,
            weight_normalization=True,
            state_include_action=False,
        )

        algo = Trainer(
            env=env,
            policy=policy,
            paths=paths,
            n_epochs=1000,
            evaluate_performance=True,
            train_ratio=0.9,
            eval_samples=10000,
            eval_horizon=1000,
            opt_batch_size=256,
            # max_path_length=1000,
            # n_eval_trajs=10,
            # eval_batch_size=10000,
            # n_eval_envs=10,
            threshold=4.,
            # batch_size=512,
            # n_slices=10,
        )

        algo.train(sess=sess)


variants = VG().variants()

print("#Experiments:", len(variants))

for v in variants:

    kwargs = dict(
        use_cloudpickle=True,
        exp_prefix="tower-bc-rnn-1",
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
            docker_image="quay.io/openai/rocky-rllab3-gpr-gpu-pascal:20170112",
            docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
        )

    run_experiment_lite(
        run_task,
        **kwargs
    )
