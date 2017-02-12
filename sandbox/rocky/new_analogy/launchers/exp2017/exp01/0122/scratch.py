from gym.spaces import prng

from sandbox.rocky.new_analogy import fetch_utils

env = fetch_utils.gpr_fetch_env(horizon=1000, height=3)


env.seed(0)
prng.seed(0)
ob = env.reset()

import ipdb; ipdb.set_trace()

