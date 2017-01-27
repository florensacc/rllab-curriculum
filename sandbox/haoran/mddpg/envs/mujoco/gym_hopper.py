import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from sandbox.haoran.mddpg.misc.mujoco_utils import convert_gym_space

class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        use_forward_reward=True,
        use_alive_bonus=True,
        use_ctrl_cost=True,
    ):
        self.use_forward_reward = use_forward_reward
        self.use_alive_bonus = use_alive_bonus
        self.use_ctrl_cost = use_ctrl_cost
        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)
        self.observation_space = convert_gym_space(self.observation_space)
        self.action_space = convert_gym_space(self.action_space)

    def _step(self, a):
        posbefore = self.model.data.qpos[0,0]
        self.do_simulation(a, self.frame_skip)
        posafter,height,ang = self.model.data.qpos[0:3,0]
        alive_bonus = 1.0
        reward = 0
        if self.use_forward_reward:
            reward += (posafter - posbefore) / self.dt
        if self.use_alive_bonus:
            reward += alive_bonus
        if self.use_ctrl_cost:
            reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        com = self.get_body_com("torso")
        return ob, reward, done, {"com": com}

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat[1:],
            np.clip(self.model.data.qvel.flat,-10,10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + np.random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + np.random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def get_param_values(self):
        params = np.array([
            self.use_forward_reward,
            self.use_alive_bonus,
            self.use_ctrl_cost,
        ])
        return params

    def set_param_values(self, params):
        self.use_forward_reward = params[0]
        self.use_alive_bonus = params[1]
        self.use_ctrl_cost = params[2]


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

        HopperEnv.plot_paths(paths, ax)

        return stats

    @staticmethod
    def plot_paths(paths, ax):
        for path in paths:
            com = path['env_infos']['com']
            xx = com[:, 0]
            zz = com[:, 2]
            ax.plot(xx, zz, 'b')
        xlim = np.ceil(np.max(np.abs(xx)))
        ax.set_xlim((-xlim, xlim))
        ax.set_ylim((0, 1))

    def terminate(self):
        pass
