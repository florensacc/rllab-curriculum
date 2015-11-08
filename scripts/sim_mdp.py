import os
from mdp.swimmer_mdp import SwimmerMDP

mdp = SwimmerMDP()
state = mdp.reset()[0]
#mdp.plot()
#state = mdp.step(state, [0])[0]

for i in range(400):
    state = mdp.step(state, [0.5, 0.5])[0]
    mdp.plot()
