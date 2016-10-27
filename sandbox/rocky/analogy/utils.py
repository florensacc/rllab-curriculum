from sandbox.rocky.tf.envs.base import TfEnv
import contextlib
import random
import numpy as np
import subprocess


def unwrap(env):
    if isinstance(env, TfEnv):
        return unwrap(env.wrapped_env)
    return env


@contextlib.contextmanager
def using_seed(seed):
    rand_state = random.getstate()
    np_rand_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    yield
    random.setstate(rand_state)
    np.random.set_state(np_rand_state)



def conopt_run_experiment(*args, **kwargs):
    from rllab import config
    from rllab.misc.instrument import run_experiment_lite
    env = kwargs.pop("env", dict())
    type_filter = kwargs.pop("type_filter")
    env = dict(
        env,
        AWS_ACCESS_KEY_ID=config.AWS_ACCESS_KEY,
        AWS_SECRET_ACCESS_KEY=config.AWS_ACCESS_SECRET,
    )
    mode = kwargs.pop("mode", "local")
    use_gpu = kwargs.pop("use_gpu", False)

    if type_filter == "pascal":
        config.DOCKER_IMAGE = "dementrock/rllab3-conopt-gpu-cuda80"
    elif type_filter == "maxwell":
        config.DOCKER_IMAGE = "dementrock/rllab3-conopt-gpu"
    else:
        raise NotImplementedError

    config.KUBE_DEFAULT_NODE_SELECTOR = {
    }
    config.KUBE_DEFAULT_RESOURCES = {
        "requests": {
            "cpu": 8,
            "memory": "10Gi",
        },
    }

    if mode == "local_docker":
        if use_gpu:
            try:
                result = subprocess.check_output("nvidia-smi")
                lines = result.decode().split("\n")
                gpu_ids = [lines[idx-1].split()[1] for idx, line in enumerate(lines) if "0MiB" in line or "23MiB" in
                           line]
                if len(gpu_ids) > 0:
                    env["CUDA_VISIBLE_DEVICES"] = gpu_ids[0]
                else:
                    env["CUDA_VISIBLE_DEVICES"] = ""
                if 'GeForce GTX' in result.decode():
                    config.DOCKER_IMAGE = "dementrock/rllab3-conopt-gpu"
                else:
                    config.DOCKER_IMAGE = "dementrock/rllab3-conopt-gpu-cuda80"
            except Exception as e:
                print(e)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
    env["DISPLAY"] = ":99"
    pre_commands = [
        "export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/mesa/:$LD_LIBRARY_PATH"
    ]

    if mode == "local":
        python_command = "python"
    else:
        python_command = 'xvfb-run -a -s "-ac -screen 0 1400x900x24 +extension RANDR" -- python'

    run_experiment_lite(
        *args,
        **kwargs,
        env=env,
        mode=mode,
        use_gpu=use_gpu,
        pre_commands=pre_commands,
        python_command=python_command,
    )

