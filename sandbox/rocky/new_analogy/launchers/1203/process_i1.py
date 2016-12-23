import os
import pickle

from rllab.misc.instrument import run_experiment_lite
from rllab import config
import numpy as np
import tarfile

MODE = "local_docker"

def run_task(*_):
    tar_path = "/shared-data/I1-5k-v3.tar.gz"

    tar = tarfile.open(tar_path, 'r')
    traj_files = tar.getmembers()

    from rllab.misc.ext import using_seed
    from conopt import envs

    exp_x = []
    exp_u = []
    exp_task_ids = []
    exp_rewards = []
    exp_cost_expansions = []

    # with using_seed(0):
    #     experiment = envs.load("I1")
    #     env = experiment.make()

    for traj_file in sorted(traj_files, key=lambda x: x.name):
        if not traj_file.name.endswith(".pkl"):
            continue
        print(traj_file.name)

        f = tar.extractfile(traj_file.name)

        traj = pickle.load(f)
        solution = traj.solution

        if np.linalg.norm(solution['x']) >= 1E4:
            print("Skipped")
            continue
        if np.linalg.norm(solution['u']) >= 1E4:
            print("Skipped")
            continue

        x = solution['x']
        new_exp_x = []
        for i in range(x.shape[0]):
            new_exp_x.append(traj.env.world.observe(x[i])[0])

        exp_x.append(new_exp_x)
        exp_u.append(solution['u'])
        exp_task_ids.append(traj.env.task_id)
        exp_rewards.append(solution['reward'])
        exp_cost_expansions.append(traj.cost_expansion)
        print(traj.env.task_id)

    out_file = "/shared-data/I1-5k-v3-data.npz"

    np.savez_compressed(
        out_file,
        exp_x=exp_x,
        exp_u=exp_u,
        exp_task_ids=exp_task_ids,
        exp_rewards=exp_rewards,
        exp_cost_expansions=exp_cost_expansions,
    )


if MODE == "local":
    env = dict(PYTHONPATH=":".join([
        config.PROJECT_PATH,
        os.path.join(config.PROJECT_PATH, "conopt_root"),
    ]))
else:
    env = dict(PYTHONPATH="/root/code/rllab:/root/code/rllab/conopt_root")

if MODE in ["local_docker"]:
    env["CUDA_VISIBLE_DEVICES"] = "1"

run_experiment_lite(
    run_task,
    use_cloudpickle=True,
    exp_prefix="process-i1",
    mode=MODE,
    use_gpu=True,
    snapshot_mode="last",
    sync_all_data_node_to_s3=False,
    n_parallel=0,
    env=env,
    docker_image="quay.io/openai/rocky-rllab3-conopt-gpu-pascal",
    docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
    variant=dict(),
    seed=0,
)
