import os
import pickle

import numpy as np

from rllab import config
from rllab.misc.instrument import VariantGenerator, variant
from rllab.misc.instrument import run_experiment_lite

"""
Test performance of new network
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [1, 2, 3]

    # @variant
    # def task_id_filter(self):
    #     yield [0, 1, 2]
    #     yield [0, 1]
    #     yield [0]


USE_GPU = True
MODE = "local_docker"
# MODE = launch_cirrascale("pascal")

vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))


def run_task(v):
    from sandbox.rocky.new_analogy.tf.algos import Trainer
    from sandbox.rocky.analogy.policies.modular_analogy_policy import ModularAnalogyPolicy
    from sandbox.rocky.new_analogy.tf.networks.tower.attention import Net
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.rocky.new_analogy.envs.gpr_env import GprEnv
    from gpr_package.bin import tower_copter_policy as tower

    task_id = tower.get_task_from_text("ab")
    env = TfEnv(GprEnv(
        "tower",
        task_id=task_id,
        experiment_args=dict(nboxes=2, horizon=100),
    ))

    policy = ModularAnalogyPolicy(
        env_spec=env.spec,
        name="policy",
        net=Net(obs_type='full_state'),
    )

    file_name = "/shared-data/tower_copter_paths_ab_crippled_100"#00"
    with open(file_name, "rb") as f:
        paths = pickle.load(f)

    for p in paths:
        p["env_infos"]["task_id"] = np.asarray(task_id)

    paths = paths[:1]
    # paths =
    # import ipdb; ipdb.set_trace()

    algo = Trainer(
        env=env,
        policy=policy,
        demo_path=None,#"/shared-data/tower_copter_paths_ab_crippled_10000",
        paths=paths,
        train_ratio=0.9,
        n_passes_per_epoch=1,
        horizon=100,
        n_epochs=1000,
        learning_rate=1e-2,
        no_improvement_tolerance=10,
        batch_size=64,
        eval_samples=10000,
        eval_horizon=100,
        threshold=4.,
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
        exp_prefix="tower-single-analogy-1",
        mode=MODE,
        n_parallel=0,
        env=env,
        seed=vv["seed"],
        snapshot_mode="last",
        variant=vv,
        terminate_machine=True,
        sync_all_data_node_to_s3=False,
        use_gpu=USE_GPU,
        docker_image="quay.io/openai/rocky-rllab3-gpr-gpu-pascal:20170112",
        docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
    )
