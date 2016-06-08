from __future__ import print_function
from __future__ import absolute_import

from sandbox.rocky.hrl.envs.seaquest_grid_world_env import SeaquestGridWorldEnv
import numpy as np


from rllab.sampler import parallel_sampler
from rllab.policies.uniform_control_policy import UniformControlPolicy
import multiprocessing as mp

parallel_sampler.initialize(mp.cpu_count())

env = SeaquestGridWorldEnv(size=5, n_bombs=1, n_divers=2)
policy = UniformControlPolicy(env_spec=env.spec)

parallel_sampler.populate_task(env, policy)

paths = parallel_sampler.sample_paths(
    policy.get_param_values(),
    max_samples=10000,
    max_path_length=100
)

returns = [np.sum(p["rewards"]) for p in paths]
# avg_return = np.mean()
# max_return = np.mean([np.sum(p["rewards"]) for p in paths])

print("Average reward: %f" % np.mean(returns))
print("Max reward: %f" % np.max(returns))
print("Min reward: %f" % np.min(returns))


# see how many scores by change

# while True:
#     env.render()
#
# # for _ in range(100):
# #     obs = env.reset()
# #     while True:
# #         action = np.random.randint(low=0, high=4)
# #         _, _, done, _ = env.step(action)
# #         if done:
# #             break
