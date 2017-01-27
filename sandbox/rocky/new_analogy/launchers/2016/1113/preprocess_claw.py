# First experiment with claw, using behavior cloning
# The end goal is to get some reasonable behavior using as few demonstrations as possible
from rllab.misc.instrument import run_experiment_lite
import pickle
import numpy as np


def process_0(*_):
    from conopt import envs

    import os
    threshold = 4.5
    path = "/local_home/rocky/claw-20k"
    traj_files = os.listdir(path)

    exp_x = []
    exp_u = []
    exp_rewards = []

    from rllab.misc.ext import using_seed
    with using_seed(0):
        experiment = envs.load("TF2")
        env = experiment.make(0)

    for traj_file in sorted(traj_files):
        print(traj_file)
        full_path = os.path.join(path, traj_file)
        with open(full_path, "rb") as f:
            traj = pickle.load(f)
            solution = traj.solution
            # if solution['reward'][-1] < threshold:
            #     continue
            #
            if np.linalg.norm(solution['x']) >= 1E4:
                continue
            if np.linalg.norm(solution['u']) >= 1E4:
                continue

            x = solution['x']
            new_exp_x = []
            for i in range(x.shape[0]):
                new_exp_x.append(env.world.observe(x[i]))
                # print(env.world.observe(x[i]))
                # sys.exit()
                # import ipdb; ipdb.set_trace()
                # pass
            exp_x.append(new_exp_x)
            exp_u.append(solution['u'])
            exp_rewards.append(solution['reward'])

    exp_x = np.asarray(exp_x)
    exp_u = np.asarray(exp_u)
    exp_rewards = np.asarray(exp_rewards)

    out_file = "/home/rocky/conopt-shared-data/claw-20k-data.npz"

    np.savez_compressed(out_file, exp_x=exp_x, exp_u=exp_u, exp_rewards=exp_rewards)

    # import ipdb;
    # ipdb.set_trace()
    # pass


def process_1(*_):
    file = "/shared-data/claw-20k-data.npz"
    data = np.load(file)
    exp_x = data["exp_x"]
    exp_u = data["exp_u"]
    exp_rewards = data["exp_rewards"]

    ids_2k = np.random.choice(np.arange(len(exp_x)), size=2000, replace=False)
    ids_500 = np.random.choice(np.arange(len(exp_x)), size=500, replace=False)

    np.savez_compressed(
        "/shared-data/claw-2k-data.npz",
        exp_x=exp_x[ids_2k], exp_u=exp_u[ids_2k], exp_rewards=exp_rewards[ids_2k]
    )
    np.savez_compressed(
        "/shared-data/claw-500-data.npz",
        exp_x=exp_x[ids_500], exp_u=exp_u[ids_500], exp_rewards=exp_rewards[ids_500]
    )


def process_2(*_):
    file = "/shared-data/claw-20k-data.npz"
    data = np.load(file)
    exp_x = data["exp_x"]
    exp_u = data["exp_u"]
    exp_rewards = data["exp_rewards"]

    for size in [100, 50, 20, 10]:
        ids = np.random.choice(np.arange(len(exp_x)), size=size, replace=False)
        np.savez_compressed(
            "/shared-data/claw-{}-data.npz".format(size),
            exp_x=exp_x[ids], exp_u=exp_u[ids], exp_rewards=exp_rewards[ids]
        )


if __name__ == "__main__":
    run_experiment_lite(
        process_2,
        use_cloudpickle=True,
        # exp_prefix="trpo-gym-6",
        # variant=v,
        mode="local_docker",  # MODE,
        use_gpu=False,
        snapshot_mode="last",
        sync_all_data_node_to_s3=False,
        n_parallel=0,
        env=dict(PYTHONPATH="/root/code/rllab:/root/code/rllab/conopt_root"),
        docker_image="quay.io/openai/rocky-rllab3-conopt-gpu-pascal",
        # docker_image="dementrock/rllab3-shared-gpu-cuda80",
        docker_args=" -v /home/rocky/conopt-shared-data:/shared-data"
        # seed=v["seed"]
    )
    # break

    # main()
