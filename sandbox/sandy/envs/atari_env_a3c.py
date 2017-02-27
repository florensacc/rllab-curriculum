"""Wrapper around AtariEnv, to make it work with A3C implementation.
"""
from sandbox.sandy.envs.atari_env import AtariEnv, get_base_env
from sandbox.sandy.async_rl.shareable.base import Shareable
from sandbox.sandy.async_rl.utils.picklable import Picklable
from sandbox.sandy.shared.ale_compatibility import set_gym_seed

class AtariEnvA3C(AtariEnv, Shareable, Picklable):
    def __init__(self, env_name, record_video=False, video_schedule=None, \
                 log_dir=None, record_log=True, force_reset=False, **kwargs):
 
        self.init_params = locals()
        self.init_params.pop('self')

        AtariEnv.__init__(self, env_name, record_video=record_video, \
                          video_schedule=video_schedule, log_dir=log_dir, \
                          record_log=record_log, force_reset=force_reset, **kwargs)

        self.legal_actions = self.base_env.ale.getMinimalActionSet()

    def set_seed(self, seed):
        set_gym_seed(self.base_env, int(seed))

    def prepare_sharing(self):
        self.share_params = dict()

    def process_copy(self):
        """
        Simply create an env with the same initialization.
        """
        import copy
        init_params_copy = copy.deepcopy(self.init_params)
        if 'kwargs' in init_params_copy:
            kwargs = init_params_copy.pop('kwargs')
            new_env = AtariEnvA3C(**init_params_copy, **kwargs)
        else:
            new_env = AtariEnvA3C(**init_params_copy)
        new_env.frame_dropout = self.frame_dropout  # In case this was changed
        return new_env

    def receive_action(self, action):
        if self.is_terminal:  # Do not step if episode has ended
            print("Not calling step because episode has ended")
            return 0
        self.step(action)
        return self._reward

    def update_params(self, global_vars, training_args):
        pass

    def initialize(self):
        self.reset()

    def init_copy_from(self, target_env):
        from sandbox.sandy.shared.model_rollout import set_seed_env
        self.gym_seed = target_env.gym_seed
        set_seed_env(self.env, self.gym_seed)
        self.initialize()
        self.phase = "Test"
        assert(self.persistent_adv == target_env.persistent_adv)

    @property
    def extra_infos(self):
        return {}

    @property
    def state(self):
        #return list(self.unscale_obs(self.observation))
        return list(self.observation)

    @property
    def number_of_actions(self):
        return len(self.legal_actions)
