import multiprocessing
import pickle

from rllab.misc.instrument import run_experiment_lite

"""
Behavior cloning on fetch using noisy trajectories
"""

MODE = "local_docker"
N_PARALLEL = 1


def run_task(vv):
    from gpr_package.bin import tower_fetch_policy as tower
    from sandbox.rocky.s3.resource_manager import resource_manager
    import subprocess
    import numpy as np

    ls_output = subprocess.check_output([
        "aws",
        "s3",
        "ls",
        "s3://rllab-rocky/resource/tower_fetch_ab/",
    ])

    resource_names = [x.split()[-1] for x in ls_output.decode().split('\n')[:-1]]

    task_id = tower.get_task_from_text("ab")
    expr = tower.SimFetch(nboxes=2, horizon=2500, mocap=True, obs_type="full_state")
    env = expr.make(task_id)
    policy = tower.FetchPolicy(task_id=task_id)
    from rllab.envs.gym_env import convert_gym_space
    obs_space = convert_gym_space(env.observation_space)

    paths = []

    for resource in resource_names:
        file_name = resource_manager.get_file("tower_fetch_ab/{}".format(resource))
        print("Loading {}".format(resource))
        with open(file_name, "rb") as f:
            paths.extend(pickle.load(f))

    with multiprocessing.Pool() as pool:
        for idx, path in enumerate(paths):
            print(idx)
            obs = obs_space.unflatten_n(path["observations"])
            actions = np.asarray(pool.map(policy.get_action, obs))
            path["taught_actions"] = actions

    resource_manager.register_data("tower_fetch_ab_annotated", pickle.dumps(paths))

kwargs = dict(
    use_cloudpickle=True,
    mode=MODE,
    use_gpu=True,
    snapshot_mode="last",
    sync_all_data_node_to_s3=False,
    n_parallel=N_PARALLEL,
    env=dict(CUDA_VISIBLE_DEVICES="4", PYTHONPATH="/root/code/rllab:/root/code/rllab/gpr_package"),
    variant=dict(),
)

if MODE == "local":
    del kwargs["env"]["PYTHONPATH"]  # =
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
