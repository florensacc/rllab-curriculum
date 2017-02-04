import tensorflow
import numpy as np
from scipy import interpolate
# from sandbox.rocky.new_analogy.algos.pi2 import PI2
np.random.multivariate_normal(np.zeros(3), np.ones((3,3)))

# from sandbox.rocky.new_analogy.algos.pi2 import PI2
#
# from gpr_package.bin import tower_fetch_policy as tower
# import numpy as np
#
# task_id = tower.get_task_from_text("ab")
# expr = tower.SimFetch(nboxes=2, horizon=500, mocap=True, obs_type="full_state")
# policy = tower.FetchPolicy(task_id)
# env = expr.make(task_id=task_id)
#
# xinit = env.world.sample_xinit()
# ob = env.reset_to(xinit)
# acts = []
# for _ in range(500):
#     action = policy.get_action(ob)
#     acts.append(action)
#     ob, reward, done, _ = env.step(action)
# print("Final reward: ", reward)
#
#
# algo = PI2(
#     env=env,
#     xinit=xinit,
#     num_iterations=100,
#     particles=100,
#     init_k=np.asarray(acts),
# )
# algo.train()
