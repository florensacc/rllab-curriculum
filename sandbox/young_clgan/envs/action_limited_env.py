from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv
import numpy as np
from rllab import spaces


class ActionLimitedEnv(ProxyEnv, Serializable):

    # limit action space and reset space for environment that can actuate objects in the environment
    def __init__(self, env,
                 motor_controlled_actions=(0,9),
                 reset_time_steps=200,
                 position_controlled_actions=None,
                 ):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self.motor_controlled_actions = motor_controlled_actions
        self.reset_time_steps = reset_time_steps
        self.position_controlled_actions = position_controlled_actions

    @property
    # by default lb = -1 x 7, 0,0 and ub = 1x7, 0, 0
    def action_space(self):
        bounds = self.model.actuator_ctrlrange
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        return spaces.Box(lb[:-2], ub[:-2])

    def reset(self, init_state=None, **kwargs):
        # ret = self.wrapped_env.reset(init_state=init_state, **kwargs)

        # finds innermost mujoco environment
        self.base_env = self
        while hasattr(self.base_env, "wrapped_env"):
            self.base_env = self.base_env.wrapped_env
        self.base_env.reset(init_state, **kwargs)

        self.base_action = [0, 0, 0, 0, 0, 0, 0] #todo: I think this does nothing, should check
        self.random_position = [0, 0.2] # can change site position to visualize
        # print(self.base_env.model.data.qpos)
        # self.random_position = np.random.uniform(-0.01, 0.01, 2)
        for i in range(self.reset_time_steps):
            self.base_env.step(np.append(self.base_action, self.random_position))


        # self.wrapped_env.reset(self, init_state, *args, **kwargs) # probably need to do self.wrapped_env
        return self.wrapped_env.get_current_obs() # todo: might not work for other environments

    def step(self, action):
        # self.random_position = [0,0]
        # print(self.base_env.model.data.qpos)
        action = np.append(action, self.random_position)
        # print("Action in wrapper:", action)
        return self.base_env.step(action)
        # return self.wrapped_env.step(action)