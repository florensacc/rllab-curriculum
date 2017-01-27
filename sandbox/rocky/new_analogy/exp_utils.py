import glob
import multiprocessing
import os
import subprocess
import random

from rllab import config
from rllab.misc.instrument import run_experiment_lite
from sandbox.rocky.cirrascale.launch_job import launch_cirrascale


def find_free_gpu():
    result = subprocess.check_output("nvidia-smi")
    lines = result.decode().split("\n")
    gpu_ids = [
        lines[idx - 1].split()[1]
        for idx, line in enumerate(lines) if "0MiB" in line or "23MiB" in line
        ]
    return gpu_ids[0]


def run_local_docker(runner, exp_name=None, variant=None, seed=None, n_parallel=None, use_gpu=True):
    if exp_name is None:
        exp_name = "experiment"
    if n_parallel is None:
        n_parallel = 1

    if use_gpu:
        env=dict(
            CUDA_VISIBLE_DEVICES=find_free_gpu(),
            PYTHONPATH="/root/code/rllab:/root/code/rllab/gpr_package"
        )
    else:
        env=dict(
            CUDA_VISIBLE_DEVICES="",
            PYTHONPATH="/root/code/rllab:/root/code/rllab/gpr_package"
        )

    run_experiment_lite(
        runner,
        use_cloudpickle=True,
        mode="local_docker",
        exp_prefix=exp_name,
        exp_name=exp_name,
        snapshot_mode="last",
        seed=seed,
        env=env,
        variant=variant,
        use_gpu=True,
        n_parallel=n_parallel,
    )


def run_cirrascale(runner, exp_name=None, variant=None, seed=None, n_parallel=None):
    if exp_name is None:
        exp_name = "experiment"
    if n_parallel == None:
        n_parallel = multiprocessing.cpu_count()
    run_experiment_lite(
        runner,
        use_cloudpickle=True,
        mode=launch_cirrascale("pascal"),
        exp_prefix=exp_name,
        snapshot_mode="last",
        seed=seed,
        env=dict(
            PYTHONPATH="/root/code/rllab:/root/code/rllab/gpr_package"
        ),
        variant=variant,
        sync_all_data_node_to_s3=False,
        terminate_machine=True,
        use_gpu=True,
        n_parallel=n_parallel,
    )


def run_local(runner, exp_name=None, variant=None, seed=None, n_parallel=None):
    if exp_name is None:
        exp_name = "experiment"
    if os.path.exists("/home/cirrascale"):
        use_gpu = True
        env = dict(
            CUDA_VISIBLE_DEVICES=find_free_gpu(),
        )
    else:
        use_gpu = False
        env = dict()
    if n_parallel is None:
        n_parallel = multiprocessing.cpu_count()

    run_experiment_lite(
        runner,
        use_cloudpickle=True,
        mode="local",
        exp_prefix=exp_name,
        exp_name=exp_name,
        snapshot_mode="last",
        seed=seed,
        env=env,
        variant=variant,
        use_gpu=use_gpu,
        n_parallel=n_parallel,
    )


def run_ec2(runner, exp_name, variant=None, seed=None, n_parallel=None, instance_type="c4.8xlarge"):
    config.AWS_INSTANCE_TYPE = instance_type
    config.AWS_SPOT = True
    config.AWS_SPOT_PRICE = '2.0'
    config.AWS_REGION_NAME = random.choice(
        ['us-west-1', 'us-east-1', 'us-west-2']
    )
    config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[config.AWS_REGION_NAME]
    config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[config.AWS_REGION_NAME]
    config.AWS_SECURITY_GROUP_IDS = config.ALL_REGION_AWS_SECURITY_GROUP_IDS[config.AWS_REGION_NAME]

    if n_parallel is None:
        n_parallel = multiprocessing.cpu_count()

    run_experiment_lite(
        runner,
        use_cloudpickle=True,
        mode="ec2",
        exp_prefix=exp_name,
        snapshot_mode="last",
        seed=seed,
        env=dict(
            PYTHONPATH="/root/code/rllab:/root/code/rllab/gpr_package"
        ),
        variant=variant,
        sync_all_data_node_to_s3=False,
        terminate_machine=True,
        use_gpu=False,
        n_parallel=n_parallel,
    )



def terminate_ec2_instances(exp_name, filter_dict):
    subprocess.check_call([
        "python",
        "scripts/sync_s3.py",
        exp_name,
    ])

    folders = glob.glob(os.path.join(config.LOG_DIR, "s3/{}/*".format(exp_name)))
    import ipdb; ipdb.set_trace()


