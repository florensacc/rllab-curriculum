import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from sandbox.haoran.mddpg.misc.mujoco_utils import convert_gym_space

class Walker2DEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)
        self.observation_space = convert_gym_space(self.observation_space)
        self.action_space = convert_gym_space(self.action_space)

    def _step(self, a):
        posbefore = self.model.data.qpos[0,0]
        self.do_simulation(a, self.frame_skip)
        posafter,height,ang = self.model.data.qpos[0:3,0]
        alive_bonus = 1.0
        reward = ((posafter - posbefore) / self.dt )
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0
                    and ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        com = self.get_body_com("torso")
        return ob, reward, done, {"com": com}

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel,-10,10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + np.random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + np.random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
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

    def log_stats(self, alg, epoch, paths, ax):
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

        Walker2DEnv.plot_paths(paths, ax)

        return stats

    @staticmethod
    def plot_paths(paths, ax):
        for path in paths:
            com = path['env_infos']['com']
            xx = com[:, 0]
            zz = com[:, 2]
            ax.plot(xx, zz, 'b')
        ax.set_xlim((np.min(xx) - 1, np.max(xx)+1))
        ax.set_ylim((-1, 2))

    def terminate(self):
        pass
