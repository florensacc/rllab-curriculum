import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from sandbox.haoran.mddpg.misc.mujoco_utils import convert_gym_space

def mass_center(model):
    mass = model.body_mass
    xpos = model.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))[0]

class HumanoidStandupEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'humanoidstandup.xml', 5)
        utils.EzPickle.__init__(self)
        self.observation_space = convert_gym_space(self.observation_space)
        self.action_space = convert_gym_space(self.action_space)

    def _get_obs(self):
        # everything by the x, y positions of the torso frame
        # not the same as com, for unknown reason
        data = self.model.data
        return np.concatenate([data.qpos.flat[2:],
                               data.qvel.flat,
                               data.cinert.flat,
                               data.cvel.flat,
                               data.qfrc_actuator.flat,
                               data.cfrc_ext.flat])

    def _step(self, a):
        self.do_simulation(a, self.frame_skip)
        pos_after = self.model.data.qpos[2][0]
            # z coordinate of the torso frame
        data = self.model.data
        uph_cost=(pos_after - 0 ) / self.model.opt.timestep

        quad_ctrl_cost =  0.1 * np.square(data.ctrl).sum()
        quad_impact_cost = .5e-6 * np.square(data.cfrc_ext).sum()
        quad_impact_cost = min(quad_impact_cost, 10)
        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1
            # reward high torso and alive
            # discourage high control or high impact force

        done = bool(False)
        com = self.get_body_com("torso")
        height = pos_after
        return (
            self._get_obs(), reward, done,
            dict(
                reward_linup=uph_cost,
                reward_quadctrl=-quad_ctrl_cost,
                reward_impact=-quad_impact_cost,
                height=height,
                com=com
            )
        )

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + np.random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + np.random.uniform(low=-c, high=c, size=self.model.nv,)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 1.0
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def get_param_values(self):
        return None

    def set_param_values(self, params):
        pass

    def log_stats(self, alg, epoch, paths, ax):
        # forward distance
        heights = np.concatenate([
            path["env_infos"]["height"]
            for path in paths
        ])

        stats = {
            'env: HeightAverage': np.mean(heights),
            'env: HeightMedian': np.median(heights),
            'env: HeightMax': np.max(heights),
            'env: HeightMin': np.min(heights),
            'env: HeightStd': np.std(heights),
        }

        HumanoidStandupEnv.plot_paths(paths, ax)

        return stats

    @staticmethod
    def plot_paths(paths, ax):
        for path in paths:
            com = path['env_infos']['com']
            xx = com[:, 0]
            zz = com[:, 2]
            ax.plot(xx, zz, 'b')
        xlim = np.ceil(np.max(np.abs(xx)) / 0.5) * 0.5
        ax.set_xlim((-xlim, xlim))
        ax.set_ylim((0, 1.2))

    def terminate(self):
        pass
