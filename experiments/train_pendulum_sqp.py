from algo import SQP, ILQG
from mdp import GripperMDP, HopperMDP, PendulumMDP

algo = SQP()
algo.train(PendulumMDP)
