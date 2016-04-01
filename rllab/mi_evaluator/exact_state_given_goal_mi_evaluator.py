from rllab.envs.compound_action_sequence_env import CompoundActionSequenceEnv
from rllab.envs.grid_world_env import GridWorldEnv


class ExactStateGivenGoalMIEvaluator(object):
    def __init__(self, env, policy):
        assert isinstance(env, CompoundActionSequenceEnv)
        assert isinstance(env.wrapped_env, GridWorldEnv)
        self.mirror_env = env
        self.policy = policy

    def predict(self, path):
        import ipdb;
        ipdb.set_trace()
