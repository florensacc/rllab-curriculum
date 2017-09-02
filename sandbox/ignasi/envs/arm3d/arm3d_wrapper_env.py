from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
import joblib
import numpy as np
import random


class RobustDiskWrapperEnv(ProxyEnv, Serializable):
    def __init__(self, env,
                 ):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

        # load policy that moves peg
        peg_policy = "/home/michael/rllab_goal_rl/data/policies/params.pkl"
        data = joblib.load(peg_policy)
        if "algo" in data:
            self.move_peg_agent = data["algo"].policy
            self.move_peg_env = data["algo"].env
        self.max_path_length = 200 # for moving peg
        self.amount_moved = 0.05

        # locates base environment (i.e. environment that tries to move disk)
        self.base_env = self
        while hasattr(self.base_env, "wrapped_env"):
            self.base_env = self.base_env.wrapped_env

    def reset(self, init_state=None, target_position = None, **kwargs):
        # #todo: remove
        # init_state = (0.387, 1.137, -2.028, -1.744, 2.029, -0.873, 1.55, 0, 0)

        self.base_env.reset(init_state, **kwargs)
        old_qpos = self.base_env.model.data.qpos

        # target position should be set by goal environment
        if target_position is not None:
            target_position = np.array((0 + random.uniform(-self.amount_moved, self.amount_moved),
                                                 0.3 + random.uniform(-self.amount_moved, self.amount_moved)))
        o = self.move_peg_env.reset(init_state=old_qpos, target_position = target_position)
        path_length = 0
        self.move_peg_agent.reset()

        while path_length < self.max_path_length:
            a, agent_info = self.move_peg_agent.get_action(o)
            next_o, r, d, env_info = self.move_peg_env.step(a)
            path_length += 1
            if d:
                break
            o = next_o
        new_qpos = self.move_peg_env.model.data.qpos

        return self.base_env.reset(init_state=new_qpos, **kwargs)






