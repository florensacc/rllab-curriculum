import numpy as np
from contextlib import contextmanager
import os.path as osp
from rllab.mdp.base import ControlMDP
from rllab.mjcapi.rocky_mjc_1_22 import MjModel, MjViewer
from rllab.misc.overrides import overrides
from rllab.misc import autoargs
import theano
import tempfile
import mako.template
import mako.lookup


MODEL_DIR = osp.abspath(
    osp.join(
        osp.dirname(__file__),
        '../../../vendor/mujoco_models/1_22'
    )
)


class MujocoMDP(ControlMDP):

    FILE = None

    @autoargs.arg('action_noise', type=float,
                  help='Noise added to the controls, which will be '
                       'proportional to the action bounds')
    def __init__(self, action_noise=0.0, file_path=None):
        # compile template
        if file_path is None:
            if self.__class__.FILE is None:
                raise "Mujoco file not specified"
            file_path = osp.join(MODEL_DIR, self.__class__.FILE)
        if file_path.endswith(".mako"):
            lookup = mako.lookup.TemplateLookup(directories=[MODEL_DIR])
            with open(file_path) as template_file:
                template = mako.template.Template(
                    template_file.read(), lookup=lookup)
            content = template.render()
            _, file_path = tempfile.mkstemp(text=True)
            with open(file_path, 'w') as f:
                f.write(content)
        self.model = MjModel(file_path)
        self.data = self.model.data
        self.viewer = None
        self.init_qpos = self.model.data.qpos
        self.init_qvel = self.model.data.qvel
        self.init_qacc = self.model.data.qacc
        self.init_ctrl = self.model.data.ctrl
        self.qpos_dim = self.init_qpos.size
        self.qvel_dim = self.init_qvel.size
        self.ctrl_dim = self.init_ctrl.size
        self.action_noise = action_noise
        if "frame_skip" in self.model.numeric_names:
            frame_skip_id = self.model.numeric_names.index("frame_skip")
            self.frame_skip = int(self.model.numeric_data.flat[frame_skip_id])
        else:
            self.frame_skip = 1
        self.dcom = None
        self.current_com = None
        self.reset()
        super(MujocoMDP, self).__init__()

    @property
    @overrides
    def observation_shape(self):
        return self.get_current_obs().shape

    @property
    @overrides
    def observation_dtype(self):
        return theano.config.floatX

    @property
    @overrides
    def action_dim(self):
        return len(self.model.data.ctrl)

    @property
    @overrides
    def action_dtype(self):
        return theano.config.floatX

    @property
    @overrides
    def action_bounds(self):
        bounds = self.model.actuator_ctrlrange
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        return lb, ub

    def reset_mujoco(self):
        self.model.data.qpos = self.init_qpos
        self.model.data.qvel = self.init_qvel
        self.model.data.qacc = self.init_qacc
        self.model.data.ctrl = self.init_ctrl

    @overrides
    def reset(self):
        self.reset_mujoco()
        self.model.forward()
        self.current_com = self.model.data.com_subtree[0]
        self.dcom = np.zeros_like(self.current_com)
        self.current_state = self.get_current_state()
        return self.get_current_state(), self.get_current_obs()

    def get_state(self, pos, vel):
        return np.concatenate([pos.reshape(-1), vel.reshape(-1)])

    def decode_state(self, state):
        qpos, qvel = np.split(state, [self.qpos_dim])
        return qpos, qvel

    def get_current_obs(self):
        return self.get_full_obs()

    def get_full_obs(self):
        data = self.model.data
        cdists = np.copy(self.model.geom_margin).flat
        for c in self.model.data.contact:
            cdists[c.geom2] = min(cdists[c.geom2], c.dist)
        obs = np.concatenate([
            data.qpos.flat,
            data.qvel.flat,
            # data.cdof.flat,
            data.cinert.flat,
            data.cvel.flat,
            # data.cacc.flat,
            data.qfrc_actuator.flat,
            data.cfrc_ext.flat,
            data.qfrc_constraint.flat,
            cdists,
            # data.qfrc_bias.flat,
            # data.qfrc_passive.flat,
            self.dcom.flat,
        ])
        return obs

    def get_obs(self, state):
        with self.set_state_tmp(state):
            return self.get_current_obs()

    def get_current_state(self):
        return self.get_state(self.model.data.qpos, self.model.data.qvel)

    def inject_action_noise(self, action):
        # generate action noise
        noise = self.action_noise * \
            np.random.normal(size=action.shape)
        # rescale the noise to make it proportional to the action bounds
        lb, ub = self.action_bounds
        noise = 0.5 * (ub - lb) * noise
        return action + noise

    def forward_dynamics(self, state, action, restore=True):
        with self.set_state_tmp(state, restore):
            self.model.data.ctrl = self.inject_action_noise(action)
            for _ in range(self.frame_skip):
                self.model.step()
            self.model.forward()
            new_com = self.model.data.com_subtree[0]
            self.dcom = new_com - self.current_com
            self.current_com = new_com
            return self.get_current_state()

    def get_viewer(self):
        if self.viewer is None:
            self.viewer = MjViewer()
            self.viewer.start()
            self.viewer.set_model(self.model)
        return self.viewer

    def plot(self):
        viewer = self.get_viewer()
        viewer.loop_once()

    def start_viewer(self):
        viewer = self.get_viewer()
        if not viewer.running:
            viewer.start()

    def stop_viewer(self):
        if self.viewer:
            self.viewer.finish()

    def set_state(self, state):
        qpos, qvel = self.decode_state(state)
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        self.model.forward()
        self.current_state = state

    @contextmanager
    def set_state_tmp(self, state, restore=True):
        if np.array_equal(state, self.current_state) and not restore:
            yield
        else:
            if restore:
                prev_pos = self.model.data.qpos
                prev_qvel = self.model.data.qvel
                prev_ctrl = self.model.data.ctrl
                prev_act = self.model.data.act
            qpos, qvel = self.decode_state(state)
            self.model.data.qpos = qpos
            self.model.data.qvel = qvel
            self.model.forward()
            yield
            if restore:
                self.model.data.qpos = prev_pos
                self.model.data.qvel = prev_qvel
                self.model.data.ctrl = prev_ctrl
                self.model.data.act = prev_act
                self.model.forward()

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def get_body_comvel(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.body_comvels[idx]

    def print_stats(self):
        super(MujocoMDP, self).print_stats()
        print "qpos dim:\t%d" % len(self.model.data.qpos)
