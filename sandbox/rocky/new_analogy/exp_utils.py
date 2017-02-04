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


def is_cirrascale():
    return os.path.exists("/home/cirrascale") or os.path.exists("/shared-data")


def wrap_runner(runner):
    def _runner(*args, **kwargs):
        import os
        os.environ['KERAS_BACKEND'] = 'tensorflow'
        from sandbox.rocky.s3.resource_manager import resource_manager
        import builtins
        if not hasattr(builtins, "profile"):
            def _profile(fn):
                return fn
            builtins.profile = _profile
        if is_cirrascale():
            if os.path.exists("/shared-data"):
                resource_manager.local_resource_path = "/shared-data/resource"
            elif os.path.exists("/home/rocky/conopt-shared-data"):
                resource_manager.local_resource_path = "/home/rocky/conopt-shared-data/resource"
        runner(*args, **kwargs)
    return _runner


def run_local_docker(runner, exp_name=None, variant=None, seed=None, n_parallel=None, use_gpu=True):
    if exp_name is None:
        exp_name = "experiment"
    if n_parallel is None:
        n_parallel = multiprocessing.cpu_count()

    if use_gpu:
        env = dict(
            CUDA_VISIBLE_DEVICES=find_free_gpu(),
            PYTHONPATH="/root/code/rllab:/root/code/rllab/gpr_package"
        )
    else:
        env = dict(
            CUDA_VISIBLE_DEVICES="",
            PYTHONPATH="/root/code/rllab:/root/code/rllab/gpr_package"
        )

    args = dict(
        use_cloudpickle=True,
        mode="local_docker",
        exp_prefix=exp_name,
        exp_name=exp_name,
        snapshot_mode="last",
        env=env,
        variant=variant,
        use_gpu=use_gpu,
        n_parallel=n_parallel,
    )

    if is_cirrascale():
        args["docker_args"] = " -v /home/rocky/conopt-shared-data:/shared-data"

    if seed is not None:
        args["seed"] = seed

    run_experiment_lite(
        wrap_runner(runner),
        **args,
    )


def run_cirrascale(runner, exp_name=None, variant=None, seed=None, n_parallel=None):
    if exp_name is None:
        exp_name = "experiment"
    if n_parallel == None:
        n_parallel = multiprocessing.cpu_count()
    run_experiment_lite(
        wrap_runner(runner),
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
        sync_log_on_termination=False,
        n_parallel=n_parallel,
        docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
    )


def run_local(runner, exp_name=None, variant=None, seed=None, n_parallel=None, use_gpu=True, profile=False):
    if exp_name is None:
        exp_name = "experiment"
    if is_cirrascale() and use_gpu:
        env = dict(
            CUDA_VISIBLE_DEVICES=find_free_gpu(),
        )
    else:
        use_gpu = False
        env = dict()
    if n_parallel is None:
        n_parallel = multiprocessing.cpu_count()

    args = dict(
        use_cloudpickle=True,
        mode="local",
        exp_prefix=exp_name,
        exp_name=exp_name,
        snapshot_mode="last",
        env=env,
        variant=variant,
        use_gpu=use_gpu,
        n_parallel=n_parallel,
    )

    if seed is not None:
        args["seed"] = seed
    if profile:
        args["python_command"] = "kernprof -l"

    run_experiment_lite(
        wrap_runner(runner),
        **args
    )


def run_ec2(
        runner,
        exp_name,
        variant=None,
        seed=None,
        n_parallel=None,
        instance_type="c4.8xlarge",
        disk_size=100,
):
    config.AWS_INSTANCE_TYPE = instance_type
    config.AWS_SPOT = True
    config.AWS_SPOT_PRICE = '2.0'
    config.AWS_REGION_NAME = random.choice(
        ['us-west-1', 'us-east-1', 'us-west-2']
    )
    config.AWS_KEY_NAME = config.ALL_REGION_AWS_KEY_NAMES[config.AWS_REGION_NAME]
    config.AWS_IMAGE_ID = config.ALL_REGION_AWS_IMAGE_IDS[config.AWS_REGION_NAME]
    config.AWS_SECURITY_GROUP_IDS = config.ALL_REGION_AWS_SECURITY_GROUP_IDS[config.AWS_REGION_NAME]
    if disk_size is not None:
        config.AWS_EXTRA_CONFIGS = dict(
            BlockDeviceMappings=[
                {
                    'DeviceName': '/dev/sda1',
                    'Ebs': {
                        'VolumeSize': disk_size,
                        'VolumeType': 'standard'
                    }
                }
            ]
        )

    if n_parallel is None:
        n_parallel = multiprocessing.cpu_count()

    run_experiment_lite(
        wrap_runner(runner),
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
        sync_log_on_termination=False,
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
    import ipdb;
    ipdb.set_trace()
