from gym.spaces import prng

from bin.tower_fetch_policy import FetchPolicy
from sandbox.rocky.new_analogy import fetch_utils
import numpy as np

# gpr_nomocap_env = fetch_utils.gpr_fetch_env(mocap=False)

# gpr_env = fetch_utils.gpr_fetch_env()
#
# demo_policy = fetch_utils.fetch_prescribed_policy(gpr_env)

from gpr.envs import fetch_rl
from sandbox.rocky.new_analogy.fetch_utils import absolute2relative_wrapper

expr = fetch_rl.Experiment(horizon=300)
env = expr.make(LL=True)

policy = FetchPolicy(env.task_id).get_action
policy = absolute2relative_wrapper(policy)

env.seed(0)
prng.seed(0)

env.reset()
for _ in range(300):
    env.render()
    env.step(policy(env))
    import time; time.sleep(0.001)

env = fetch_utils.fetch_env(horizon=50)
path = fetch_utils.demo_path(env=env, animated=True)
trajectory = fetch_utils.path_to_traj(gpr_env=env.wrapped_env.gpr_env, path=path)

import ipdb; ipdb.set_trace()

import pickle
with open("/tmp/trajectory.pkl", "wb") as f:
    pickle.dump(trajectory, f)


# ob = gpr_env.reset()
# gpr_nomocap_env.reset_to(gpr_env.x)
#
# robot_joint_names = list(np.asarray(gpr_env.world.model.joint_names)[gpr_env.world._get_robot_indices()])
# actuator_names = gpr_nomocap_env.world.model.actuator_names
# actuator_ids = [robot_joint_names.index(x) for x in gpr_nomocap_env.world.model.actuator_names]
#
# while True:
#     x_before = gpr_env.x
#     a = demo_policy.get_action(ob)[0]
#     next_ob, _, _, _ = gpr_env.step(a)
#
#     ctrl = next_ob[0][actuator_ids]
#     gpr_nomocap_env.reset_to(x_before)
#     for _ in range(10):
#         ob, _, _, _ = gpr_nomocap_env.step(ctrl)
#     import ipdb; ipdb.set_trace()
#     gpr_env.reset_to(gpr_nomocap_env.x)
#     gpr_nomocap_env.render()
#     import time; time.sleep(0.002)
#
#     # path = fetch_utils.demo_traj(seed=0)

    # import ipdb;
    #
    # ipdb.set_trace()
