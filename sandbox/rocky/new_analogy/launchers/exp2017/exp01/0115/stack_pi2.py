from rllab.misc import logger
from rllab.misc.instrument import run_experiment_lite, variant, VariantGenerator

"""
Behavior clone single trajectory
"""

MODE = "local_docker"  # _docker"  # _docker"
# MODE = launch_cirrascale("pascal")
N_PARALLEL = 1  # 8


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31]  # , 41, 51]


def run_task(vv):
    import numpy as np

    from gpr.envs.stack import Experiment
    expr = Experiment(nboxes=2, horizon=500)
    env = expr.make(task_id=[[1, "top", 0]])
    env.seed(vv["seed"])
    np.random.seed(vv["seed"])

    xinit = env.world.sample_xinit()

    from sandbox.rocky.new_analogy.tf.algos import PI2
    logger.log("Launching PI2...")
    algo = PI2(
        env=env,
        xinit=xinit,
        num_iterations=15,
        particles=500,
        init_cov=1.,
    )
    algo.train()


variants = VG().variants()

print("#Experiments:", len(variants))

for v in variants:

    kwargs = dict(
        use_cloudpickle=True,
        exp_prefix="tower-pi2",
        exp_name="tower-pi2",
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
            docker_image="quay.io/openai/rocky-rllab3-gpr-gpu-pascal:20170114",
            docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
        )

    run_experiment_lite(
        run_task,
        **kwargs,
        # pre_commands=["pip install --upgrade numpy"]
    )
