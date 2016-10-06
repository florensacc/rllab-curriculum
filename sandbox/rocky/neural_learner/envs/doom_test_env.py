from sandbox.rocky.neural_learner.envs.doom_default_wad_env import DoomDefaultWadEnv


class DoomTestEnv(DoomDefaultWadEnv):

    def __init__(self, *args, **kwargs):
        DoomDefaultWadEnv.__init__(
            self,
            wad_name="/tmp/b765f3d8-02f4-40c6-9125-ddd7c65a7dbd.wad",
            *args, **kwargs
        )