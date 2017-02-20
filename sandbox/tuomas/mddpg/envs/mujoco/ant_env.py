from sandbox.tuomas.mddpg.envs.mujoco.mujoco_env import MujocoEnv
from sandbox.haoran.myscripts.quaternion import Quaternion
from rllab.core.serializable import Serializable
import numpy as np

from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger


class AntEnv(MujocoEnv, Serializable):

    FILE = 'ant.xml'

    def __init__(self,
            direction=None,
            reward_type="velocity",
            reset_penalty=None,
            leg_zpos_thr=2.0,
            flip_thr=0.,
            *args,
            **kwargs
        ):
        super(AntEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())
        if direction is not None:
            assert np.isclose(np.linalg.norm(direction), 1.)
        self.direction = direction
        self.reward_type = reward_type
        self.reset_penalty = reset_penalty
        self.leg_zpos_thr = leg_zpos_thr
        self.flip_thr = flip_thr

    def get_param_values(self):
        params = dict(
            direction=self.direction,
            reward_type=self.reward_type,
            reset_penalty=self.reset_penalty,
            leg_zpos_thr=self.leg_zpos_thr,
            flip_thr=self.flip_thr,
        )
        return params

    def set_param_values(self, params):
        self.__dict__.update(params)

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)

    def step(self, action):
        self.forward_dynamics(action)
        comvel = self.get_body_comvel("torso")
        if self.reward_type == "velocity":
            if self.direction is not None:
                motion_reward = comvel[0:2].dot(np.array(self.direction))
            else:
                motion_reward = np.linalg.norm(comvel[0:2])
        elif self.reward_type == "distance_from_origin":
            motion_reward = np.linalg.norm(self.get_body_com("torso")[:2])
        elif self.reward_type == "forward_distance":
            motion_reward = self.get_body_com("torso")[0]
        else:
            raise NotImplementedError

        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5

        action_violations = np.maximum(np.maximum(lb - action, action - ub), 0)
        action_violation_cost = np.sum((action_violations / scaling)**2)

        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        survive_reward = 0.05
        reward = (motion_reward - ctrl_cost - contact_cost + survive_reward
                  - action_violation_cost)
        state = self._state
        leg_zpos = [
            self.get_body_com("front_left_leg")[2],
            self.get_body_com("front_right_leg")[2],
            self.get_body_com("back_leg")[2],
            self.get_body_com("right_back_leg")[2],
        ]
        # q describes the orientation of the ball
        q = Quaternion(*tuple(self.model.data.qpos[3:7].ravel()))
        # z is the z-pos of the bottom of the ball after applying q
        z = q.rotate(np.array([0., 0., -1.]))[2]

        notdone = all([
            np.isfinite(state).all(),
            state[2] <= 1.0, # prevent jumpping
            # np.max(leg_zpos) < self.leg_zpos_thr,
            z < self.flip_thr, # prevent flipping
        ])
        done = not notdone

        if self.reset_penalty:
            done = False
            reward -= self.reset_penalty
        ob = self.get_current_obs()

        return Step(ob, float(reward), done, com=self.get_body_com("torso"))
        #return Step(ob, float(reward), done)

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular(
            'env: ForwardProgressAverage', np.mean(progs))
        logger.record_tabular(
            'env: ForwardProgressMax', np.max(progs))
        logger.record_tabular(
            'env: ForwardProgressMin', np.min(progs))
        logger.record_tabular(
            'env: ForwardProgressStd', np.std(progs))

    def log_stats(self, algo, epoch, paths):
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
        }

        return stats
