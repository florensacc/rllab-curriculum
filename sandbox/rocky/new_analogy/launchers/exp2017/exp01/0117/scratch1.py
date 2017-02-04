from gym.spaces import prng

import gpr_package.bin.tower_fetch_policy as tower
from gpr.envs.fetch.sim_fetch import Experiment as SimFetch

T = 3000
task_id = tower.get_task_from_text("abcde")

expr = SimFetch(nboxes=5, horizon=T, mocap=True, obs_type="full_state", num_substeps=1)
env = expr.make(task_id)

# obtain demonstration
policy = tower.FetchPolicy(task_id)
env.seed(0)
prng.seed(0)
xinit = env.world.sample_xinit()
xs = []
acts = []
ob = env.reset_to(xinit)
obs = []
rewards = []
for t in range(T):
    a = policy.get_action(ob)
    xs.append(env.x)
    acts.append(a)
    obs.append(env.observation_space.flatten(ob))

    ob, reward, _, _ = env.step(a)
    env.render()
    import time

    time.sleep(0.01)
    rewards.append(reward)
