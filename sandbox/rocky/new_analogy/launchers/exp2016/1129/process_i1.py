import os
import pickle

from rllab.misc.instrument import run_experiment_lite
from rllab import config
import numpy as np

MODE = "local_docker"


# MODE = launch_cirrascale("pascal")


def run_task(*_):
    path = "/shared-data/copter-2-targets"

    traj_files = os.listdir(path)

    from rllab.misc.ext import using_seed
    from conopt import envs

    exp_x = []
    exp_u = []
    exp_task_ids = []
    exp_rewards = []

    with using_seed(0):
        experiment = envs.load("I1")
        env = experiment.make()

    for traj_file in sorted(traj_files):
        if not traj_file.endswith(".pkl"):
            continue
        print(traj_file)
        full_path = os.path.join(path, traj_file)

        with open(full_path, "rb") as f:
            traj = pickle.load(f)
            traj.compute_additional_fields()
            solution = traj.solution

            import ipdb; ipdb.set_trace()

            if np.linalg.norm(solution['x']) >= 1E4:
                print("Skipped")
                continue
            if np.linalg.norm(solution['u']) >= 1E4:
                print("Skipped")
                continue

            x = solution['x']
            new_exp_x = []
            for i in range(x.shape[0]):
                new_exp_x.append(env.world.observe(x[i]))

            exp_x.append(new_exp_x)
            exp_u.append(solution['u'])
            exp_task_ids.append(traj.env.task_id)
            exp_rewards.append(solution['reward'])

    out_file = "/shared-data/copter-5k-data.npz"

    np.savez_compressed(
        out_file,
        exp_x=exp_x,
        exp_u=exp_u,
        exp_task_ids=exp_task_ids,
        exp_rewards=exp_rewards
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
