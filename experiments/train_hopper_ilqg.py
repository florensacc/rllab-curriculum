from algo import ILQG
from mdp import GripperMDP, HopperMDP, PendulumMDP

algo = ILQG()
algo.train(HopperMDP)
