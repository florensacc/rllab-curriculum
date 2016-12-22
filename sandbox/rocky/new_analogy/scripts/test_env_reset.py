import numpy as np
from rllab.misc.instrument import run_experiment_lite, variant, VariantGenerator
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale


def run_task(*_):
    from sandbox.rocky.new_analogy.envs.conopt_env import ConoptEnv
    from sandbox.rocky.tf.envs.base import TfEnv

    data = np.load("/shared-data/claw-{n_trajs}-data.npz".format(n_trajs=500))
    exp_x = data["exp_x"]
    xinits = exp_x[:, 0, :]
    xinit = xinits[0]
    FixedResetClawEnv
    env = TfEnv(ConoptEnv("TF2"))
    conopt_env = env.wrapped_env.conopt_env
    conopt_env.reset_to(xinit[:conopt_env.world.dimx])
    import ipdb; ipdb.set_trace()



run_experiment_lite(
    run_task,
    use_cloudpickle=True,
    exp_prefix="gail-finetune-claw-2",
    mode="local_docker",
    # mode=launch_cirrascale("pascal"),
    use_gpu=False,#True,
    snapshot_mode="last",
    sync_all_data_node_to_s3=False,
    n_parallel=8,
    env=dict(PYTHONPATH="/root/code/rllab:/root/code/rllab/conopt_root"),
    docker_image="quay.io/openai/rocky-rllab3-conopt-gpu-pascal",
    docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
    variant=dict(),
    seed=0,#v["seed"],
)
