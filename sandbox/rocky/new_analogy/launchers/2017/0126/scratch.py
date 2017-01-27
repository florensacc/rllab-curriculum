import pyprind

from rllab.misc import logger
from rllab.sampler.utils import rollout
from sandbox.rocky.new_analogy.exp_utils import run_local_docker, run_cirrascale, run_ec2, run_local
from sandbox.rocky.new_analogy.policies.auto_mlp_policy import AutoMLPPolicy
from sandbox.rocky.tf.policies.categorical_mlp_policy import CategoricalMLPPolicy

"""
Dagger but with multiple initial positions, for two blocks
"""


def run_task(vv):
    from sandbox.rocky.new_analogy import fetch_utils
    import numpy as np

    n_boxes = vv["n_boxes"]

    if n_boxes == 5:
        horizon = 2000
    else:
        raise NotImplementedError

    n_configurations = 100

    disc_intervals = np.asarray([
        [-0.3, -0.1, -0.03, -0.01, -0.003, -0.001, -0.0003, 0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
        [-0.3, -0.1, -0.03, -0.01, -0.003, -0.001, -0.0003, 0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
        [-0.3, -0.1, -0.03, -0.01, -0.003, -0.001, -0.0003, 0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    ])

    env = fetch_utils.fetch_env(horizon=horizon, height=n_boxes)
    demo_policy = fetch_utils.fetch_discretized_prescribed_policy(env=env, disc_intervals=disc_intervals)

    logger.log("Generating initial demonstrations")
    _paths = fetch_utils.new_policy_paths(
        seeds=np.random.randint(low=0, high=np.iinfo(np.int32).max, size=n_configurations),
        env=env,
        policy=demo_policy,
        # noise_levels=[0., 1e-3, 1e-2]
    )
    logger.log("Generated")

    # filter out failed paths
    stage_init_pts = []
    paths = []
    for p in _paths:
        pts = fetch_utils.find_stageinit_points(env, p)
        if len(pts) == n_boxes:
            stage_init_pts.append(pts)
            paths.append(p)

    stage_xinits = []
    for pts, path in zip(stage_init_pts, paths):
        stage_xinits.extend(path["env_infos"]["x"][pts[:-1]])

    # use stage as the first index
    stage_xinits = np.asarray(stage_xinits)

    import ipdb; ipdb.set_trace()


run_local_docker(
    run_task,
    variant=dict(
        n_boxes=5,
    ),
    seed=0,
    n_parallel=0,
)
