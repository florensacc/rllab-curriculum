import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from sandbox.haoran.mddpg.misc.mujoco_utils import convert_gym_space

class UndirectedSwimmerEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
    Typical swimmer, but with a constant reward dist_reward given if the
        the x coordinate >= dist_threshold.
    """
    def __init__(
        self,
        dist_threshold=4.,
        dist_reward=1.,
    ):
        self.dist_threshold = dist_threshold
        self.dist_reward = dist_reward
        mujoco_env.MujocoEnv.__init__(self, 'swimmer.xml', 4)
        utils.EzPickle.__init__(self)
        self.observation_space = convert_gym_space(self.observation_space)
        self.action_space = convert_gym_space(self.action_space)

    def _step(self, a):
        ctrl_cost_coeff = 0.0001
        # xposbefore = self.model.data.qpos[0,0]
        self.do_simulation(a, self.frame_skip)
        # xposafter = self.model.data.qpos[0,0]
        # reward_vel = np.abs(xposafter - xposbefore) / self.dt
        xposafter = self.get_body_com("torso")[0]
        reward_vel = np.linalg.norm(self.get_body_comvel("torso")[:2])
        reward_ctrl = - ctrl_cost_coeff * np.square(a).sum()
        reward_dist = self.dist_reward * (xposafter > self.dist_threshold)
        reward = reward_vel + reward_ctrl + reward_dist
        ob = self._get_obs()
        return ob, reward, False, dict(
                reward_vel=reward_vel,
                reward_ctrl=reward_ctrl,
                reward_dist=reward_dist,
                pos=self.model.data.qpos[0:2,0],
                com=self.get_body_com("torso")[:2],
            )

    def _get_obs(self):
        qpos = self.model.data.qpos
        qvel = self.model.data.qvel
        return np.concatenate([qpos.flat[2:], qvel.flat])

    def reset_model(self):
        self.set_state(
            self.init_qpos + np.random.uniform(low=-.1, high=.1, size=self.model.nq),
            self.init_qvel + np.random.uniform(low=-.1, high=.1, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def get_param_values(self):
        return dict()

    def set_param_values(self, params):
        pass

    def log_stats(self, alg, epoch, paths, ax):
        # forward distance
        progs = []
        for path in paths:
            # pos = path["env_infos"]["pos"]
            pos = path["env_infos"]["com"]
            progs.append(pos[-1][0] - pos[0][0])
            # x-coord of com at the last time step minus the 1st step

        # log vel reward and dist reward
        rewards_vel = np.concatenate([path["env_infos"]["reward_vel"]])
        rewards_dist = np.concatenate([path["env_infos"]["reward_dist"]])

        stats = {
            'env: ForwardProgressAverage': np.mean(progs),
            'env: ForwardProgressMax': np.max(progs),
            'env: ForwardProgressMin': np.min(progs),
            'env: VelRewardAverage': np.mean(rewards_vel),
            'env: VelRewardMax': np.max(rewards_vel),
            'env: VelRewardMin': np.min(rewards_vel),
            'env: DistRewardAverage': np.mean(rewards_dist),
            'env: DistRewardMax': np.max(rewards_dist),
            'env: DistRewardMin': np.min(rewards_dist),
        }

        UndirectedSwimmerEnv.plot_paths(paths, ax)

        return stats

    @staticmethod
    def plot_paths(paths, ax):
        for path in paths:
            pos = path['env_infos']['pos']
            xx = pos[:, 0]
            yy = pos[:, 1]
            ax.plot(xx, yy, 'b')
        xlim = np.ceil(np.max(np.abs(xx)))
        ax.set_xlim((-xlim, xlim))
        ax.set_ylim((-4, 4))

    def terminate(self):
        pass
