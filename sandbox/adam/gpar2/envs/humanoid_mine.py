"""
This is like Humanoid-v1, the gym environment, except this will skip gym and be
an rllab mujoco environment.

Will be the first one implemented with pre- allcating mujoco_py.

Maybe I should double check that this runs, first...OK it does.
"""

# from rllab.envs.mujoco.mujoco_env import MujocoEnv
# Just to point it to my own mujoco_py version:
from sandbox.adam.gpar.envs.mujoco_env_mine import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
# from rllab.misc import logger
from ctypes import POINTER, c_double  # needed for setting controls faster
from rllab import spaces

BIG = 1e6

# Old way.
# def mass_center(model):
#     mass = model.body_mass
#     xpos = model.data.xipos
#     return (np.sum(mass * xpos, 0) / np.sum(mass))[0]


class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


class HumanoidEnv(MujocoEnv, Serializable):

    FILE = 'humanoid.xml'

    def __init__(
            self,
            vel_deviation_cost_coeff=1e-2,
            alive_bonus=0.2,
            ctrl_cost_coeff=1e-3,
            impact_cost_coeff=1e-5,
            *args, **kwargs):
        self.vel_deviation_cost_coeff = vel_deviation_cost_coeff
        self.alive_bonus = alive_bonus
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.impact_cost_coeff = impact_cost_coeff
        super(HumanoidEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

        self.frame_skip = 5  # For some reason this was getting set differently.
        print("frame skip: ", self.frame_skip)
        self.preallocate()
        self.prepare_spaces()
        self.reset()  # call this AFTER preallocate(), uses it.

    def preallocate(self):
        """
        write all the variables that will be used for storing results from the
        model data, and used for calculating rewards, etc.

        All are flat unless denoted as shaped.
        """
        self.mass = self.model.body_mass.squeeze()
        self.mass_sum_inv = 1 / np.sum(self.mass)

        data = self.model.data
        alloc = dict2()

        xipos = data.xipos
        alloc.xipos = np.zeros(xipos.size)  # make it write to contiguous memory
        alloc.xipos_shaped = alloc.xipos.reshape(xipos.shape)  # different view to same memory

        alloc.ctrl = np.zeros(data.ctrl.size)
        alloc.ctrl_set = np.zeros(alloc.ctrl.size)
        alloc.ctrl_set_ptr = alloc.ctrl_set.ctypes.data_as(POINTER(c_double))
        alloc.cfrc_ext = np.zeros(data.cfrc_ext.size)

        # Prepare which values to get for the obersvation, and where they fit into
        # the final observation vector returned at each step.
        self.obs_list = ["qpos", "qvel", "cinert", "cvel", "qfrc_actuator", "cfrc_ext"]
        for obs in self.obs_list:
            alloc[obs] = np.zeros(getattr(data, obs).size)
            # print(obs, "size: ", getattr(data, obs).size)

        # Values for reset.
        alloc.init_qpos = data.qpos.squeeze()
        alloc.init_qpos_rand = np.zeros(alloc.init_qpos.shape)
        alloc.init_qpos_ptr = alloc.init_qpos_rand.ctypes.data_as(POINTER(c_double))
        alloc.init_qvel = data.qvel.squeeze()
        alloc.init_qvel_rand = np.zeros(alloc.init_qvel.shape)
        alloc.init_qvel_ptr = alloc.init_qvel_rand.ctypes.data_as(POINTER(c_double))
        alloc.init_qacc = data.qacc.squeeze()
        alloc.init_qacc_ptr = alloc.init_qacc.ctypes.data_as(POINTER(c_double))
        alloc.init_ctrl = data.ctrl.squeeze()
        alloc.init_ctrl_ptr = alloc.init_ctrl.ctypes.data_as(POINTER(c_double))

        self.alloc = alloc

    def prepare_spaces(self):
        bounds = self.model.actuator_ctrlrange
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        self._action_space = spaces.Box(lb, ub)

        shp = self.get_current_obs_pre().shape
        ub = BIG * np.ones(shp)
        self._observation_space = spaces.Box(ub * -1, ub)

    @property
    @overrides
    def action_space(self):
        return self._action_space

    @property
    @overrides
    def observation_space(self):
        return self._observation_space

    # New, fast way.
    def mass_center(self):
        self.model.data.xipos_pre(self.alloc.xipos)
        return self.alloc.xipos_shaped[:, 0].dot(self.mass) * self.mass_sum_inv

    # Borrowed from mujoco_env_gym
    def do_simulation(self, ctrl, n_frames):
        # self.model.data.ctrl = ctrl  # Old way
        self.alloc.ctrl_set[:] = ctrl  # New way
        self.model.data.ctrl_set_ptr(self.alloc.ctrl_set_ptr)
        for _ in range(n_frames):
            self.model.step()

    # Old Way
    def get_current_obs(self):
        data = self.model.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    # New, hopefully fast way.
    def get_current_obs_pre(self):
        """
        This way, using concatenate, is faster than writing to a pre-allocated
        observations array and returning a copy of it.
        """
        data = self.model.data
        alloc = self.alloc
        # This for loop is nice for the code but it's faster when written out.
        # for obs in self.obs_list:
        #     getattr(data, obs + "_pre")(alloc[obs])
        data.qpos_pre(alloc.qpos)
        data.qvel_pre(alloc.qvel)
        data.cinert_pre(alloc.cinert)
        data.cvel_pre(alloc.cvel)
        data.qfrc_actuator_pre(alloc.qfrc_actuator)
        data.cfrc_ext_pre(alloc.cfrc_ext)
        return np.concatenate([alloc.qpos[2:],
                               alloc.qvel,
                               alloc.cinert,
                               alloc.cvel,
                               alloc.qfrc_actuator,
                               alloc.cfrc_ext])

    def step(self, action):
        # pos_before = self.mass_center()  # New
        # pos_before = mass_center(self.model)  # Old
        self.do_simulation(action, self.frame_skip)
        current_obs = self.get_current_obs_pre()  # populates everything in self.obs_list, and returns a copy
        pos_after = self.mass_center()  # New
        # pos_after = mass_center(self.model)  # Old
        alive_bonus = 5.0
        lin_vel_cost = 0.25 * (pos_after - self.pos_before) / self.model.opt.timestep
        self.pos_before = pos_after
        # quad_ctrl_cost = 0.1 * np.square(self.model.data.ctrl).sum()  # OLD
        # quad_impact_cost = .5e-6 * np.square(self.model.data.cfrc_ext).sum()  # OLD
        quad_ctrl_cost = 0.1 * np.square(self.model.data.ctrl_pre(self.alloc.ctrl)).sum()  # NEW. hmm, this could just use action, unless mujoco has some internal tempering. yeah leave it.
        quad_impact_cost = .5e-6 * np.square(self.alloc.cfrc_ext).sum()  # NEW. updated in get_obs
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = lin_vel_cost - quad_ctrl_cost - quad_impact_cost + alive_bonus

        done = bool((self.alloc.qpos[2] < 1.0) or (self.alloc.qpos[2] > 2.0))  # equivalent to use alloc.qpos
        return current_obs, reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost, reward_alive=alive_bonus, reward_impact=-quad_impact_cost)

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        self.pos_before = self.mass_center()  # New
        return self.get_current_obs_pre()

    @overrides
    def reset(self, init_state=None):
        self.reset_mujoco(init_state)
        # self.model.forward()  # appears unused
        # self.current_com = self.model.data.com_subtree[0]  # appears unused
        # self.dcom = np.zeros_like(self.current_com)  # appears unused
        self.pos_before = self.mass_center()
        return self.get_current_obs_pre()

    @overrides
    def reset_mujoco(self, init_state=None):
        data = self.model.data
        alloc = self.alloc
        if init_state is None:
            alloc.init_qpos_rand[:] = alloc.init_qpos + \
                np.random.normal(size=alloc.init_qpos.shape) * 0.01
            alloc.init_qvel_rand[:] = alloc.init_qvel + \
                np.random.normal(size=alloc.init_qvel.shape) * 0.1
            # init_qacc and init_ctrl not modified.
        else:
            start = 0
            num = alloc.init_qpos.size
            alloc.init_qpos_rand[:] = init_state[start:start + num]
            start += num
            num = alloc.init_qvel.size
            alloc.init_qvel_rand[:] = init_state[start:start + num]
            start += num
            num = alloc.init_qacc.size
            alloc.init_qacc[:] = init_state[start:start + num]
            start += num
            num = alloc.init_ctrl.size
            alloc.init_ctrl[:] = init_state[start:start + num]

        data.qpos_set_ptr(alloc.init_qpos_ptr)
        data.qvel_set_ptr(alloc.init_qvel_ptr)
        data.qacc_set_ptr(alloc.init_qacc_ptr)
        data.ctrl_set_ptr(alloc.init_ctrl_ptr)

    # @overrides
    # def log_diagnostics(self, paths):
    #     progs = [
    #         path["observations"][-1][-3] - path["observations"][0][-3]
    #         for path in paths
    #     ]
    #     logger.record_tabular('AverageForwardProgress', np.mean(progs))
    #     logger.record_tabular('MaxForwardProgress', np.max(progs))
    #     logger.record_tabular('MinForwardProgress', np.min(progs))
    #     logger.record_tabular('StdForwardProgress', np.std(progs))
