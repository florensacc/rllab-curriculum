from rllab.algo.ilqg import ILQG
from rllab.mdp.swimmer_mdp import SwimmerMDP

mdp = SwimmerMDP()
mdp.horizon = 20#50
ILQG().train(mdp)
