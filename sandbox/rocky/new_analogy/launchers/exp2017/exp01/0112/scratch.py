import os
import pickle

from rllab import config
from rllab.misc import logger
from rllab.misc.instrument import VariantGenerator, variant
from rllab.misc.instrument import run_experiment_lite
from sandbox.rocky.new_analogy.tf.policies.residual_policy import ResidualPolicy
from sandbox.rocky.s3.resource_manager import resource_manager

"""
Train copter task using TRPO. Fixed to a single task_id
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31, 41, 51]#, 2, 3]


USE_GPU = True
MODE = "local"#_docker"#launch_cirrascale("pascal")

vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))


def run_task(v):
    from sandbox.rocky.tf.algos.trpo import TRPO
    from sandbox.rocky.tf.policies.gaussian_mlp_policy import GaussianMLPPolicy
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv
    from gpr_package.bin import tower_copter_policy as tower
    from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
    from sandbox.rocky.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline

    logger.log("Loading data...")
    file_name = resource_manager.get_file("tower_copter_paths_ab_100")
    with open(file_name, 'rb') as f:
        paths = pickle.load(f)

    import ipdb; ipdb.set_trace()

    logger.log("Loaded")
    path = paths[0]

    task_id = tower.get_task_from_text("ab")

    env = TfEnv(GprEnv(
        "tower",
        task_id=task_id,
        experiment_args=dict(nboxes=2, horizon=1000),
        xinits=[paths[0]["env_infos"]["x"][800]],
    ))

    # for t, x in enumerate(path['env_infos']['x']):

    # env.wrapped_env.gpr_env.reset_to(path['env_infos']['x'][900])
    # env.render()
    #     # import time
    #     # time.sleep(0.05)
    #     # print(t)
    # import ipdb; ipdb.set_trace()

    policy = ResidualPolicy(env_spec=env.spec, wrapped_policy=GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(128, 128),
        name="policy",
    ))

    # policy = GaussianMLPPolicy(
    #     env_spec=env.spec,
    #     hidden_sizes=(128, 128),
    #     name="policy",
    # )

    baseline = GaussianMLPBaseline(
        env_spec=env.spec,
        regressor_args=dict(
            use_trust_region=True,
            hidden_sizes=(128, 128),
            optimizer=ConjugateGradientOptimizer(),
            step_size=0.1,
        ),
    )

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=10000,
        max_path_length=100,
        discount=0.99,
        gae_lambda=0.97,
        step_size=0.01,
        n_itr=5000,
    )

    algo.train()


if MODE == "local":
    env = dict(PYTHONPATH=":".join([
        config.PROJECT_PATH,
        os.path.join(config.PROJECT_PATH, "gpr_package"),
    ]))
else:
    env = dict(PYTHONPATH="/root/code/rllab:/root/code/rllab/gpr_package")

if MODE in ["local_docker"]:
    env["CUDA_VISIBLE_DEVICES"] = "1"

for vv in variants:
    run_experiment_lite(
        run_task,
        use_cloudpickle=True,
        exp_prefix="scratch",
        mode=MODE,
        n_parallel=0,
        env=env,
        seed=vv["seed"],
        snapshot_mode="last",
        variant=vv,
        terminate_machine=True,
        sync_all_data_node_to_s3=False,
        use_gpu=USE_GPU,
        # docker_image="quay.io/openai/rocky-rllab3-gpr-gpu-pascal:20170111",
        # docker_args=" -v /home/rocky/gpr-shared-data:/shared-data",
    )
