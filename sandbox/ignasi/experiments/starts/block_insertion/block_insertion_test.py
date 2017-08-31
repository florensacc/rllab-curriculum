from sandbox.young_clgan.envs.block_insertion.block_insertion_env import BLOCK_INSERTION_ENVS
import numpy as np

inner_env = BLOCK_INSERTION_ENVS[0]()
inner_env.reset()
inner_env.render(close=False)
while True:
    inner_env.step(np.array([0]))
    pass
