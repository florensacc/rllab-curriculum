from gym.spaces import prng

from sandbox.rocky.new_analogy import fetch_utils
import numpy as np

env = fetch_utils.fetch_env(horizon=2000, height=5)

gpr_env = fetch_utils.get_gpr_env(env)

gpr_env.seed(0)
prng.seed(0)

ob1 = gpr_env.reset()
print(gpr_env.step(np.zeros(env.action_space.flat_dim))[0])
print(gpr_env.step(np.zeros(env.action_space.flat_dim))[0])

vec_env = env.vec_env_executor(n_envs=1)

ob1_ = vec_env.reset(dones=[True], seeds=[0])[0]
print(vec_env.step([np.zeros(env.action_space.flat_dim)])[0][0])
print(vec_env.step([np.zeros(env.action_space.flat_dim)])[0][0])

# print(ob1)
# print(ob1_)


