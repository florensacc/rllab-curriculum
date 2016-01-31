from rllab.mdp.box2d.cartpole_mdp import CartpoleMDP
from rllab.mdp.normalized_mdp import normalize
import numpy as np

mdp = normalize(CartpoleMDP())

action = np.ones(1)
state, obs = mdp.reset()
import time
t = time.time()

for i in xrange(10000):
    mdp.step(state, action)

print time.time() - t
