from rllab import spaces
from rllab.envs.mujoco.mujoco_env import BIG
from rllab.envs.base import Env, Step
from rllab.mujoco_py import MjModel, MjViewer
import numpy as np

xml = """

<mujoco model="obstacle_course">
    <compiler inertiafromgeom="true" angle="radian" coordinate="local" />
    <option timestep="0.01" gravity="0 0 0" iterations="20" integrator="Euler" />
    <default>
        <joint limited="false" damping="1" />
        <geom contype="2" conaffinity="1" condim="1" friction=".5 .1 .1" density="1000" margin="0.002" />
    </default>
  <asset>
    <texture type="skybox" builtin="gradient" width="100" height="100" rgb1="1 1 1" rgb2="0 0 0" />
    <texture name="texgeom" type="cube" builtin="flat" mark="cross" width="127" height="1278" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" markrgb="1 1 1" random="0.01" />
    <texture name="texplane" type="2d" builtin="checker" rgb1="0 0 0" rgb2="0.8 0.8 0.8" width="100" height="100" />
    <material name='MatPlane' texture="texplane" shininess="1" texrepeat="30 30" specular="1"  reflectance="0.5" />
    <material name='geom' texture="texgeom" texuniform="true" />
  </asset>
    <worldbody>
        <light directional="true" cutoff="100" exponent="1" diffuse="1 1 1" specular=".1 .1 .1" pos="0 0 1.3" dir="-0 0 -1.3" />
    <geom name='floor' material="MatPlane" pos='0 0 -0.1' size='40 40 0.1' type='plane' conaffinity='1' rgba='0.8 0.9 0.8 1' condim='3' />
        <body name="particle" pos="0 0 0" >
            <geom name="particle_geom" type="capsule" fromto="-0.01 0 0 0.01 0 0" size="0.1" />
            <site name="particle_site" pos="0 0 0" size="0.01" />
            <joint name="ball_x" type="slide" pos="0 0 0" axis="1 0 0" armature="0"/>
            <joint name="ball_y" type="slide" pos="0 0 0" axis="0 1 0" armature="0"/>
        </body>
        <body name="target" pos="1 0 0">
            <geom name="target_geom" type="capsule" fromto="-0.01 0 0 0.01 0 0" size="0.1" rgba="0 0.9 0.1 1"/>
        </body>
    </worldbody>

    <actuator>
        <motor joint="ball_x" ctrlrange="-1.0 1.0" gear="10" ctrllimited="true"/>
        <motor joint="ball_y" ctrlrange="-1.0 1.0" gear="10" ctrllimited="true"/>
    </actuator>
</mujoco>
"""


class ForceFieldEnv(Env):
    def __init__(self):
        self.model = MjModel(xml_string=xml)
        self.viewer = None
        self.data = self.model.data
        self.viewer = None
        self.init_qpos = self.model.data.qpos
        self.init_qvel = self.model.data.qvel
        self.init_qacc = self.model.data.qacc
        self.init_ctrl = self.model.data.ctrl
        self.qpos_dim = self.init_qpos.size
        self.qvel_dim = self.init_qvel.size
        self.ctrl_dim = self.init_ctrl.size
        if "frame_skip" in self.model.numeric_names:
            frame_skip_id = self.model.numeric_names.index("frame_skip")
            addr = self.model.numeric_adr.flat[frame_skip_id]
            self.frame_skip = int(self.model.numeric_data.flat[addr])
        else:
            self.frame_skip = 1
        if "init_qpos" in self.model.numeric_names:
            init_qpos_id = self.model.numeric_names.index("init_qpos")
            addr = self.model.numeric_adr.flat[init_qpos_id]
            size = self.model.numeric_size.flat[init_qpos_id]
            init_qpos = self.model.numeric_data.flat[addr:addr + size]
            self.init_qpos = init_qpos
        self.dcom = None
        self.current_com = None
        self.reset()
        super(ForceFieldEnv, self).__init__()

    @property
    def action_space(self):
        bounds = self.model.actuator_ctrlrange
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        return spaces.Box(lb, ub)

    @property
    def observation_space(self):
        shp = self.get_current_obs().shape
        ub = BIG * np.ones(shp)
        return spaces.Box(ub * -1, ub)

    @property
    def action_bounds(self):
        return self.action_space.bounds

    def reset_mujoco(self, init_state=None):
        if init_state is None:
            self.model.data.qpos = self.init_qpos + \
                                   np.random.normal(size=self.init_qpos.shape) * 0.01
            self.model.data.qvel = self.init_qvel + \
                                   np.random.normal(size=self.init_qvel.shape) * 0.1
            self.model.data.qacc = self.init_qacc
            self.model.data.ctrl = self.init_ctrl
        else:
            start = 0
            for datum_name in ["qpos", "qvel", "qacc", "ctrl"]:
                datum = getattr(self.model.data, datum_name)
                datum_dim = datum.shape[0]
                datum = init_state[start: start + datum_dim]
                setattr(self.model.data, datum_name, datum)
                start += datum_dim

    def reset(self, init_state=None):
        self.reset_mujoco(init_state)
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        return self.get_current_obs()

    def get_current_obs(self):
        return self._get_full_obs()

    def _get_full_obs(self):
        data = self.model.data
        cdists = np.copy(self.model.geom_margin).flat
        for c in self.model.data.contact:
            cdists[c.geom2] = min(cdists[c.geom2], c.dist)
        obs = np.concatenate([
            data.qpos.flat,
            data.qvel.flat,
        ])
        return obs

    @property
    def _state(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat
        ])

    @property
    def _full_state(self):
        return np.concatenate([
            self.model.data.qpos,
            self.model.data.qvel,
            self.model.data.qacc,
            self.model.data.ctrl,
        ]).ravel()

    def forward_dynamics(self, action):
        self.model.data.ctrl = action
        for _ in range(self.frame_skip):
            self.model.step()
        self.model.forward()
        new_com = self.model.data.com_subtree[0]
        self.dcom = new_com - self.current_com
        self.current_com = new_com

    def get_viewer(self):
        if self.viewer is None:
            self.viewer = MjViewer()
            self.viewer.start()
            self.viewer.set_model(self.model)
        return self.viewer

    def render(self):
        viewer = self.get_viewer()
        viewer.loop_once()

    def start_viewer(self):
        viewer = self.get_viewer()
        if not viewer.running:
            viewer.start()

    def stop_viewer(self):
        if self.viewer:
            self.viewer.finish()

    def release(self):
        # temporarily alleviate the issue (but still some leak)
        from rllab.mujoco_py.mjlib import mjlib
        mjlib.mj_deleteModel(self.model._wrapped)
        mjlib.mj_deleteData(self.data._wrapped)

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def get_body_comvel(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.body_comvels[idx]

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        print(next_obs)

        reward = 0  # forward_reward - ctrl_cost
        done = False
        # print 'obs x: {}, obs y: {}'.format(next_obs[-3],next_obs[-2])
        return Step(next_obs, reward, done)


if __name__ == "__main__":
    env = ForceFieldEnv()
    env.reset()
    env.start_viewer()
    while True:
        env.render()
        env.step(env.action_space.sample())
