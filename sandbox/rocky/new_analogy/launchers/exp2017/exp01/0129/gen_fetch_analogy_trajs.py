import itertools

import joblib

from rllab.misc import logger
from sandbox.rocky.new_analogy.exp_utils import run_local_docker, run_cirrascale, run_ec2, run_local
from sandbox.rocky.s3 import resource_manager

"""
Try without dagger, new obs space
"""


def run_task(vv):
    from sandbox.rocky.new_analogy import fetch_utils
    from bin.tower_copter_policy import get_task_from_text
    import numpy as np

    horizon = 500
    n_configurations = vv["n_configurations"]

    noise_levels = [0., 1e-3, 1e-2]

    n_boxes = 5

    xinits_file_name = resource_manager.get_file("fetch_1000_xinits_{}_boxes.pkl".format(n_boxes))
    xinits = joblib.load(xinits_file_name)

    logger.log("Generating initial demonstrations")
    task_id_str = vv["task_id_str"]
    task_id = get_task_from_text(task_id_str)
    logger.log("Generating demonstrations for task {}".format(task_id_str))
    env = fetch_utils.fetch_env(horizon=horizon, height=n_boxes, task_id=task_id)
    demo_policy = fetch_utils.fetch_discretized_prescribed_policy(env=env, disc_intervals=fetch_utils.disc_intervals)
    paths = fetch_utils.new_policy_paths(
        seeds=np.random.randint(low=0, high=np.iinfo(np.int32).max, size=n_configurations),
        env=env,
        policy=demo_policy,
        noise_levels=noise_levels,
        xinits=xinits,
    )
    file_name = resource_manager.tmp_file_name(file_ext="pkl")
    joblib.dump(paths, file_name, compress=3)
    resource_manager.register_file("fetch_analogy_paths/task_{}_trajs_{}.pkl".format(task_id_str,
                                                                                     n_configurations), file_name)
    logger.log("Generated")


all_task_ids = list(map("".join, itertools.permutations("abcde", 2)))
for n_configurations in [100, 1000]:
    for task_id_str in all_task_ids:
        run_cirrascale(
            # run_local(
            run_task,
            exp_name="gen-fetch-analogy-trajs-1",
            variant=dict(task_id_str=task_id_str, n_configurations=n_configurations),
            seed=0,
            n_parallel=0,
            # use_gpu=False,
        )
