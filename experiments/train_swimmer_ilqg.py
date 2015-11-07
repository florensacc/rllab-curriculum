from algo.ilqg import ILQG
from mdp.swimmer_mdp import SwimmerMDP

mdp = SwimmerMDP()
mdp.horizon = 50
ILQG().train(mdp)#SwimmerMDP())
