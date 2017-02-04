import multiprocessing
import os
import pickle
import numpy as np

from rllab.misc.console import mkdir_p
from rllab.misc.instrument import run_experiment_lite

env_ = [None]
root_path = "/local_home/rocky/conopt-local-data/stack_ab_trajs_v2"
processed_root_path = "/local_home/rocky/conopt-local-data/stack_ab_trajs_v2_processed"


def fast_set_state(self, state):
    if env_[0] is None:
        print("Creating env")
        from gpr.env import Env
        env_[0] = Env(**state['env'])
    state['env'] = env_[0]
    state['world'] = state['env'].world
    self.__dict__.update(state)


def process_traj(traj_file):
    try:
        from gpr import Trajectory
        Trajectory.__setstate__ = fast_set_state

        if "traj" not in traj_file:
            return
        traj_id = traj_file.split('_')[-1].split('.')[0]

        processed_path = os.path.join(processed_root_path, "path_{0}.pkl".format(traj_id))
        if not os.path.exists(processed_path):
            print(traj_file)
            full_path = os.path.join(root_path, traj_file)
            with open(full_path, "rb") as f:
                traj = pickle.load(f)
            process(traj, processed_path)
    except Exception as e:
        print(e)


def process(traj, processed_path):
    # get observations
    observations = []
    # print("Getting obs")
    for x in traj.solution["x"]:
        obs = traj.env.world.observe(x)[0]
        observations.append(obs)
    # print("Got obs")
    path = dict(
        observations=np.asarray(observations),
        actions=np.asarray(traj.solution["u"]),
        rewards=np.asarray(traj.solution["reward"]).flatten(),
        env_infos=dict(x=traj.solution["x"])
    )
    with open(processed_path, "wb") as f:
        pickle.dump(path, f)


def run_task(*_):
    # from gpr.env import Env
    mkdir_p(processed_root_path)
    traj_files = os.listdir(root_path)
    print(len(traj_files))

    with multiprocessing.Pool() as pool:
        pool.map(process_traj, traj_files)


run_task()
# run_experiment_lite(
#     run_task,
#     use_cloudpickle=True,
#     mode="local_docker",
#     # exp_prefix="stack-pi2-3",
#     # variant=dict(worker_id=worker_id),
#     # terminate_machine=True,
#     docker_args=" -v /home/rocky/conopt-shared-data:/shared-data",
#     env=dict(PYTHONPATH="/root/code/rllab:/root/code/rllab/gpr_package"),
#     # sync_log_on_termination=False,
#     # periodic_sync=False,
#     # sync_all_data_node_to_s3=False,
# )
