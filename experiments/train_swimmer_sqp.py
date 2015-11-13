from algo.sqp import SQP
from mdp.swimmer_mdp import SwimmerMDP

mdp = SwimmerMDP()
mdp.horizon = 50
SQP().train(mdp)
