import os

from rllab import config

from sandbox.rocky.cirrascale.launch_job import launch_cirrascale

from rllab.misc.instrument import VariantGenerator, variant
from rllab.misc.instrument import run_experiment_lite

"""
Copter with cost shaping
"""


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [1, 2, 3]

    @variant
    def use_shaped_cost(self):
        return [True, False]


USE_GPU = True
# MODE = "local_docker"
MODE = launch_cirrascale("pascal")

vg = VG()

variants = vg.variants()

print("#Experiments: %d" % len(variants))


def run_task(v):
    from sandbox.rocky.new_analogy.algos.shaped_cost_trainer import ShapedCostTrainer
    from sandbox.rocky.analogy.policies.modular_analogy_policy import ModularAnalogyPolicy
    from sandbox.rocky.analogy.networks.conopt_particle.double_rnn import Net
    from sandbox.rocky.tf.envs.base import TfEnv
    from sandbox.rocky.new_analogy.envs.conopt_env import ConoptEnv

    env = TfEnv(ConoptEnv("I1_copter_3_targets"))

    policy = ModularAnalogyPolicy(
        env_spec=env.spec,
        name="policy",
        net=Net(obs_type='full_state'),
    )

    algo = ShapedCostTrainer(
        env=env,
        policy=policy,
        demo_path="/shared-data/I1-3k-data.npz",
        train_ratio=0.9,
        n_passes_per_epoch=1,
        horizon=100,
        n_epochs=1000,
        learning_rate=1e-2,
        no_improvement_tolerance=10,
        batch_size=64,
        eval_samples=10000,
        eval_horizon=100,
        use_shaped_cost=v["use_shaped_cost"],
    )

    algo.train()


if MODE == "local":
    env = dict(PYTHONPATH=":".join([
        config.PROJECT_PATH,
        os.path.join(config.PROJECT_PATH, "conopt_root"),
    ]))
else:
    env = dict(PYTHONPATH="/root/code/rllab:/root/code/rllab/conopt_root")

if MODE in ["local_docker"]:
    env["CUDA_VISIBLE_DEVICES"] = "1"

for vv in variants:
    run_experiment_lite(
        run_task,
        use_cloudpickle=True,
        exp_prefix="copter-3",
        mode=MODE,
        n_parallel=0,
        env=env,
        seed=vv["seed"],
        snapshot_mode="last",
        variant=vv,
        terminate_machine=True,
        sync_all_data_node_to_s3=False,
        use_gpu=USE_GPU,
        docker_image="quay.io/openai/rocky-rllab3-conopt-gpu-pascal:latest",
        docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
    )
