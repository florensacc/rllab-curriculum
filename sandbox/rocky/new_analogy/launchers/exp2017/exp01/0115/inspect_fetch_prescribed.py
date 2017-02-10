from gpr_package.bin import tower_fetch_policy as tower
import numpy as np

horizon = 1000
num_substeps = 1
task_id = tower.get_task_from_text("ab")
expr = tower.SimFetch(nboxes=2, horizon=horizon, mocap=True, obs_type="full_state", num_substeps=num_substeps)
policy = tower.FetchPolicy(task_id)
env = expr.make(task_id=task_id)

xinit = env.world.sample_xinit()
ob = env.reset_to(xinit)
acts = []
xs = []
total_reward = 0
for _ in range(horizon):
    action = policy.get_action(ob)
    acts.append(action)
    xs.append(env.x)
    ob, reward, done, _ = env.step(action)
    total_reward += reward

stds = np.std(acts, axis=0)
print(stds)

xinit = env.world.sample_xinit()
ob = env.reset_to(xinit)
acts = []
xs = []
total_reward = 0
for _ in range(horizon):
    action = policy.get_action(ob)
    action += 0.1 * np.random.normal(size=len(action)) * stds
    acts.append(action)
    xs.append(env.x)
    ob, reward, done, _ = env.step(action)
    total_reward += reward

    env.render()
    import time

    time.sleep(0.002)

# import ipdb; ipdb.set_trace()
