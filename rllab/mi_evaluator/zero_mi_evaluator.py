from rllab.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from rllab.envs.grid_world_env import GridWorldEnv
from rllab.policies.subgoal_policy import SubgoalPolicy
from rllab.spaces.discrete import Discrete
from sandbox.rocky.grid_world_hrl_utils import ExactComputer
from rllab.misc import tensor_utils
from rllab import hrl_utils
from rllab.misc import logger
import numpy as np


class ZeroMIEvaluator(object):
    def __init__(self, env_spec, policy):
        pass

    def predict(self, path):
        return np.zeros_like(path["rewards"])

    def log_diagnostics(self, paths):
        pass

    def fit(self, paths):
        pass

