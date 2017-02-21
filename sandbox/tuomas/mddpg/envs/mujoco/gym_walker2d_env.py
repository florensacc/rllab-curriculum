import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.misc.overrides import overrides

#from sandbox.haoran.mddpg.misc.mujoco_utils import convert_gym_space

class Walker2DEnv(MujocoEnv, Serializable):

    def __init__(self, alive_bonus=1, velocity_coeff=1, v2=True):
        self.alive_bonus = alive_bonus
        self.velocity_coeff = velocity_coeff
        self.v2 = v2

        if v2:
            Walker2DEnv.FILE = 'walker2d-v2.xml'
        else:
            Walker2DEnv.FILE = 'walker2d.xml'

        super(Walker2DEnv, self).__init__()
        Serializable.quick_init(self, locals())

        #self.observation_space = convert_gym_space(self.observation_space)
        #self.action_space = convert_gym_space(self.action_space)


    @overrides
    def step(self, a):
        posbefore = self.model.data.qpos[0, 0]
        self.forward_dynamics(a)
        #self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.model.data.qpos[0:3, 0]

        left_thigh_ang, right_thigh_ang = self.model.data.qpos[[3, 6]]

        # alive_bonus = 1.0
        dt = self.model.opt.timestep
        reward = self.velocity_coeff * ((posafter - posbefore) / dt)
        reward += self.alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        if self.v2:
            done = not (0.5 < height < 2.0
                        and -2 < left_thigh_ang < 2
                        and -2 < right_thigh_ang < 2)
        else:
            done = not (height > 0.8 and height < 2.0
                        and ang > -1.0 and ang < 1.0)

        ob = self.get_current_obs()
        com = self.get_body_com("torso")
        return ob, reward, done, {"com": com}

    @overrides
    def get_current_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel,-10,10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos,
            self.init_qvel,
        )
        #self.set_state(
        #    self.init_qpos + np.random.uniform(low=-.005, high=.005, size=self.model.nq),
        #    self.init_qvel + np.random.uniform(low=-.005, high=.005, size=self.model.nv)
        #)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass

    def log_stats(self, alg, epoch, paths):
        # forward distance
        progs = []
        for path in paths:
            coms = path["env_infos"]["com"]
            progs.append(coms[-1][0] - coms[0][0])
            # x-coord of com at the last time step minus the 1st step

        stats = {
            'env: ForwardProgressAverage': np.mean(progs),
            'env: ForwardProgressMax': np.max(progs),
            'env: ForwardProgressMin': np.min(progs),
            'env: ForwardProgressStd': np.std(progs),
            'env: ForwardProgressDiff': np.max(progs) - np.min(progs),
        }

        # Walker2DEnv.plot_paths(paths, ax)

        return stats

    @staticmethod
    def plot_paths(paths, ax):
        line_lst = []
        for path in paths:
            com = path['env_infos']['com']
            xx = com[:, 0]
            zz = com[:, 2]
            line_lst += ax.plot(xx, zz, 'b')
        ax.set_xlim((np.min(xx) - 1, np.max(xx)+1))
        ax.set_ylim((-1, 2))

        return line_lst

    def terminate(self):
        pass
