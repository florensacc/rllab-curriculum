from algo.sqp import SQP
from mdp.cartpole_mdp import CartpoleMDP

mdp = CartpoleMDP()
mdp.horizon = 50
SQP().train(mdp)
