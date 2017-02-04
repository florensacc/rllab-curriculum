# sample some trajectories, and find intermediate points
# only demonstrate up to certain stage?
import pickle

import time

from gym.spaces import prng

from rllab.sampler.utils import rollout
from sandbox.rocky.new_analogy import fetch_utils
import numpy as np

from sandbox.rocky.s3 import resource_manager
from sandbox.rocky.tf.samplers.vectorized_sampler import VectorizedSampler

env = fetch_utils.fetch_env(horizon=1000, height=3, seed=0)
prng.seed(0)

intervals = np.asarray([
    [-0.3, -0.1, -0.03, -0.01, -0.003, -0.001, -0.0003, 0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    [-0.3, -0.1, -0.03, -0.01, -0.003, -0.001, -0.0003, 0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
    [-0.3, -0.1, -0.03, -0.01, -0.003, -0.001, -0.0003, 0, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
])

print(intervals.shape)

# demo_policy = fetch_utils.fetch_prescribed_policy(env)#, intervals)
demo_policy = fetch_utils.fetch_discretized_prescribed_policy(env, intervals)

path = rollout(env, demo_policy, animated=True, speedup=10)


# vec_sampler = VectorizedSampler(env=env, policy=demo_policy, n_envs=1)
# vec_sampler.start_worker()
# # demo_policy.inform_vec_env(vec_sampler.vec_env)
# path_ = vec_sampler.obtain_samples(max_path_length=1000, max_n_trajs=1, batch_size=1000, seeds=np.arange(1))
# path_ = path_[0]
# import ipdb; ipdb.set_trace()
#
# import ipdb; ipdb.set_trace()

# paths = fetch_utils.new_policy_paths(seeds=np.arange(100), policy=demo_policy, env=env)
#
# print(np.mean([env.wrapped_env._is_success(p) for p in paths]))

import ipdb;

ipdb.set_trace()
#
#
# f_name = resource_manager.tmp_file_name(file_ext="pkl")

# with open(f_name, "wb") as f:
#     pickle.dump(paths, f)
#
#
# resource_manager.register_file("fetch_3_blocks_100_demo_paths.pkl", f_name)
#
# print(min([len(p["rewards"]) for p in paths]))
file_name = resource_manager.get_file("fetch_3_blocks_100_demo_paths.pkl")

with open(file_name, "rb") as f:
    paths = pickle.load(f)

actions = np.concatenate([p["actions"] for p in paths], axis=0)

import ipdb;

ipdb.set_trace()

import matplotlib.pyplot as plt

for idx in range(actions.shape[1]):
    # plt.subplot(2, 4, idx + 1)
    actions_i = actions[:, idx]
    lb = np.percentile(actions_i, q=1)
    ub = np.percentile(actions_i, q=99)
    actions_i = np.asarray([x for x in actions_i if lb <= x <= ub])
    # print(np.median(actions_i))
    # np.percentile(actions[:, idx], q=0.01)
    # import ipdb; ipdb.set_trace()
    plt.hist(actions_i, 100)
    plt.show()

# Generate demonstrations from quantized policy


import ipdb;

ipdb.set_trace()


# def find_completion_points(env, path):
#     site_xpos = path["env_infos"]["site_xpos"]
#     # assume that there are two additional sites, stall_mocap and grip
#     n_geoms = (site_xpos.shape[1] - 6) / 3
#
#     geom_xpos = site_xpos[:, 6:].reshape((-1, n_geoms, 3)).transpose((1, 0, 2))
#
#     block_order = env.wrapped_env.gpr_env.task_id[0]
#
#     geom_xpos = geom_xpos[block_order]
#
#     pts = []
#
#     for xpos0, xpos1 in zip(geom_xpos, geom_xpos[1:]):
#         reached = np.logical_and(
#             np.logical_and(
#                 np.abs(xpos1[:, 0] - xpos0[:, 0]) < 0.02,
#                 np.abs(xpos1[:, 1] - xpos0[:, 1]) < 0.02
#             ),
#             np.abs(xpos1[:, 2] - 0.05 - xpos0[:, 2]) < 0.005
#         )
#         candidates = np.where(reached)[0]
#         if len(pts) > 0:
#             candidates = [x for x in candidates if x >= pts[-1]]
#         if len(candidates) > 0:
#             pts.append(candidates[0])
#         else:
#             break
#     return pts
#
#
# failed = 0
# for path in paths:
#     try:
#         print(find_completion_points(env, path))
#     except Exception as e:
#         import ipdb; ipdb.set_trace()
#         failed += 1
#
# print(failed)


# proposed training procedure:
# 1. collect some demonstrations of the entire procedure
# 2. extract intermediate waypoints, to be used later as initial states
# 3. only learn from
# 3. while training: for on-policy, sample both stagewise and complete trajectories (stagewise for now)
# for x in path["env_infos"]["x"]:
#     env.wrapped_env.gpr_env.reset_to(x)# = x
#     env.render()
#     time.sleep(0.01)

# diffs1 = np.abs(geom0_xpos - geom1_xpos)
# diffs2 = np.abs(geom1_xpos - geom2_xpos)
#
#
# import ipdb; ipdb.set_trace()
