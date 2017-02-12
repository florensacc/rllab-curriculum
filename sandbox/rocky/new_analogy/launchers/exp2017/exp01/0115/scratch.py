
import numpy as np
from gpr_package.bin import tower_fetch_policy as tower

task_id = tower.get_task_from_text("ab")
horizon = 1000
num_substeps = 1
expr = tower.SimFetch(nboxes=2, horizon=horizon, mocap=True, obs_type="full_state", num_substeps=num_substeps)
policy = tower.FetchPolicy(task_id)
env = expr.make(task_id=task_id)

xinit = env.world.sample_xinit()

env.seed(0)
ob = env.reset_to(xinit)

xs = [env.x]
acts = []

for t in range(horizon):
    a = policy.get_action(ob)
    ob, _, _, _ = env.step(a)
    xs.append(env.x)
    acts.append(a)

acts = np.asarray(acts)

expr = tower.SimFetch(nboxes=2, horizon=horizon, mocap=False, obs_type="full_state", num_substeps=num_substeps)
policy = tower.FetchPolicy(task_id)
env = expr.make(task_id=task_id)


x0 = xs[0]
x1 = xs[1]


dt = 0.002
qpos0, qvel0 = np.split(x0, 2)
qpos1, qvel1 = np.split(x1, 2)
qacc = (qvel1 - qvel0) / dt

env.world.model.compute_inverse(qpos0, qvel0, qacc)#sample_xinit

# given two adjacent x

import ipdb; ipdb.set_trace()
