import os
os.environ['CGT_COMPAT_MODE'] = 'theano'
os.environ['THEANO_FLAGS'] = 'device=cpu'
from mdp.cartpole_mdp import CartpoleMDP

mdp = CartpoleMDP()
state = mdp.reset()[0]
print mdp.step(state, [0])[0]
