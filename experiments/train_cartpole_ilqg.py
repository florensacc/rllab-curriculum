from rllab.algo.ilqg import ILQG
from rllab.mdp.cartpole_mdp import CartpoleMDP

mdp = CartpoleMDP()
mdp.horizon = 30
ILQG().train(mdp)
