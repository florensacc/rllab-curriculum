

from rllab.envs.base import Env


class SupervisedEnv(Env):

    def generate_training_paths(self):
        """
        :rtype: list[dict]
        """
        raise NotImplementedError

    def test_mode(self):
        raise NotImplementedError

