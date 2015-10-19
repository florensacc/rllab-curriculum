from algo import CEM#SQP, ILQG
from mdp import GripperMDP, HopperMDP, PendulumMDP

algo = CEM()
algo.train(PendulumMDP)
