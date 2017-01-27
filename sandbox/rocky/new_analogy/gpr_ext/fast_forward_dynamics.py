import multiprocessing
from copy import copy

import math

import gpr.reward
import numpy as np
import functools
import operator

from gpr.utils.rotation import mat2euler_batch, euler2mat_batch
from gpr.worldgen.world import is_in_rl, is_in_pi2
from sandbox.rocky.new_analogy.gpr_ext.rewards import SenseDistReward
from sandbox.rocky.new_analogy.pymj import MjParallelLite


def collect_specs(reward):
    if hasattr(reward, '__cached_specs'):
        return reward.__cached_specs
    if isinstance(reward, gpr.reward.DistReward):
        specs = {"use_site_xpos"}
    elif isinstance(reward, gpr.reward.PenaltyReward):
        if reward.element == "active_contacts_efc_pos":
            if reward.metric == "L2":
                specs = {"use_active_contacts_efc_pos_L2"}
            else:
                raise NotImplementedError
        elif reward.element == "qfrc_actuator":
            specs = {"use_qfrc_actuator"}
        elif reward.element == "qacc":
            specs = {"use_qacc"}
        elif reward.element == "qvel":
            specs = set()
        else:
            import ipdb;
            ipdb.set_trace()
            raise NotImplementedError
    elif isinstance(reward, gpr.reward.SequenceReward):
        spec_list = []
        for r, cond, _ in reward.seq:
            spec_list.append(collect_specs(r))
            spec_list.append(collect_specs(cond))
        specs = set.union(*spec_list)
    elif isinstance(reward, gpr.reward.AddReward):
        spec_list = [collect_specs(x) for x in reward.rewards]
        specs = set.union(*spec_list)
    elif isinstance(reward, gpr.reward.BinaryReward):
        specs = collect_specs(reward.a).union(collect_specs(reward.b))
    elif isinstance(reward, gpr.reward.NegReward):
        specs = collect_specs(reward.a_reward)
    elif isinstance(reward, gpr.reward.LessCond):
        specs = collect_specs(reward.a).union(collect_specs(reward.b))
    elif isinstance(reward, SenseDistReward):
        specs = {"use_{0}".format(reward.key)}
    elif isinstance(reward, gpr.reward.ZeroReward):
        specs = set()
    elif isinstance(reward, (int, float)):
        return set()
    elif isinstance(reward, gpr.reward.DisplacementReward):
        return {"use_site_xpos"}
    elif isinstance(reward, gpr.reward.GripReward):
        return set()  # {}#["use_joint_qpos"}
    elif isinstance(reward, gpr.reward.CoordinateReward):
        assert reward.field.startswith("site_") and reward.field.endswith("_xpos")
        return {"use_site_xpos"}
    else:
        import ipdb;
        ipdb.set_trace()
    reward.__cached_specs = specs
    return specs


def resolve_site_xpos(mjparallel, site_xpos, name):
    name = name[len("site_"):-len("_xpos")]
    idx = mjparallel.site_names.index(name)
    assert idx >= 0
    return site_xpos[..., idx * 3:(idx + 1) * 3]


def resolve_site_xmat(mjparallel, site_xmat, name):
    name = name[len("site_"):-len("_xmat")]
    idx = mjparallel.site_names.index(name)
    assert idx >= 0
    return site_xmat[..., idx * 9:(idx + 1) * 9]


def resolve_joint_qpos(mjparallel, qpos, name):
    name = name[len("joint_"):-len("_qpos")]
    joint_name_idx = mjparallel.joint_names.index(name)
    joint_idx = mjparallel.model.jnt_dofadr[joint_name_idx]
    assert joint_idx >= 0
    return qpos[..., joint_idx:joint_idx + 1]


def compute_reward(reward, sense, mjparallel):
    def residual2reward_n(x, metric):
        if metric == "L2":
            return np.sum(np.square(x), axis=1)
        elif metric == "L1":
            return np.sum(np.abs(x), axis=1)
        elif metric.startswith("GPS:"):  # log(d^2+alpha)
            params = metric[len("GPS:"):].split(",")
            assert (len(params) == 3)
            w_L2, w_log, alpha = map(float, params)
            d2 = np.sum(np.square(x), axis=1)
            return w_L2 * d2 + w_log * np.log(d2 + alpha)
        elif metric.startswith("log:"):
            params = metric[len("log:"):].split(",")
            assert (len(params) == 1)
            alpha, = map(float, params)
            d2 = np.sum(np.square(x), axis=1)
            return np.log(d2 + alpha)
        else:
            assert False, "Undefined metric %s" % metric

    def _compute_reward(reward):
        if isinstance(reward, gpr.reward.AddReward):
            return functools.reduce(operator.add, map(_compute_reward, reward.rewards))
        elif isinstance(reward, gpr.reward.ClipReward):
            a = _compute_reward(reward.a_reward)
            clip_to = np.abs(_compute_reward(reward.b_reward))
            return np.clip(a, -clip_to, clip_to)
        elif isinstance(reward, gpr.reward.NegReward):
            return -_compute_reward(reward.a_reward)
        elif isinstance(reward, gpr.reward.DistReward):
            a = resolve_site_xpos(mjparallel, sense["site_xpos"], reward.a)
            b = resolve_site_xpos(mjparallel, sense["site_xpos"], reward.b)
            x = a - b
            if reward.offset is not None:
                x -= reward.offset.reshape(1,-1)
            if reward.weights is not None:
                x *= reward.weights
            return residual2reward_n(x, reward.metric)
        elif isinstance(reward, gpr.reward.BinaryReward):
            return reward.f(_compute_reward(reward.a), _compute_reward(reward.b))
        elif isinstance(reward, gpr.reward.PenaltyReward):
            if reward.element == "active_contacts_efc_pos" and reward.metric == "L2":
                return sense["active_contacts_efc_pos_L2"]
            elif reward.element in ["qfrc_actuator", "qacc", "qvel"]:
                return residual2reward_n(sense[reward.element], reward.metric)
            else:
                import ipdb;
                ipdb.set_trace()
        elif isinstance(reward, gpr.reward.SequenceReward):
            batch_size = list(sense.values())[0].shape[0]
            ret = np.zeros(batch_size)
            filled = np.zeros(batch_size, dtype=np.bool)
            assert len(reward.seq) > 0
            for r, cond, _ in reward.seq:
                cond_val = _compute_reward(cond)
                r_val = _compute_reward(r)
                mask = np.logical_and(np.logical_not(filled), np.logical_not(cond_val))
                filled = np.logical_or(filled, np.logical_not(cond_val))
                ret[mask] = r_val[mask]
            mask = np.logical_not(filled)
            ret[mask] = r_val[mask]
            return ret
        elif isinstance(reward, gpr.reward.LessCond):
            return np.less(_compute_reward(reward.a), _compute_reward(reward.b))
        elif isinstance(reward, SenseDistReward):
            return residual2reward_n(sense[reward.key] - reward.target[None, :], reward.metric)
        elif isinstance(reward, gpr.reward.ZeroReward):
            batch_size = list(sense.values())[0].shape[0]
            return np.zeros(batch_size)
        elif isinstance(reward, gpr.reward.DisplacementReward):
            if reward.target is None:
                reward.target = np.copy(resolve_site_xpos(mjparallel, sense["site_xpos"], reward.xpos))
            diff = resolve_site_xpos(mjparallel, sense["site_xpos"], reward.xpos) - reward.target
            return residual2reward_n(diff, reward.metric)
        elif isinstance(reward, gpr.reward.GripReward):
            gripper_state = resolve_joint_qpos(mjparallel, sense["qpos"], "joint_robot:l_gripper_finger_joint_1_qpos") + \
                            resolve_joint_qpos(mjparallel, sense["qpos"], "joint_robot:r_gripper_finger_joint_1_qpos")
            return residual2reward_n(gripper_state - reward.target_value, "L2")
        elif isinstance(reward, gpr.reward.CoordinateReward):
            assert reward.field.startswith("site_") and reward.field.endswith("_xpos")
            return resolve_site_xpos(mjparallel, sense["site_xpos"], reward.field)[..., reward.coordinate]
        elif isinstance(reward, (int, float)):
            return reward
        else:
            import ipdb;
            ipdb.set_trace()

    return _compute_reward(reward)


class FastForwardDynamics(object):
    def __init__(self, env, n_parallel=None, extra_specs=set(), custom_reward=None, custom_specs=None):
        if n_parallel is None:
            n_parallel = multiprocessing.cpu_count()
        xml = env.world.xml
        xml = xml.replace('<mujoco_extended>', '')
        xml = xml.replace('</mujoco_extended>', '')
        self.mjparallel = MjParallelLite(
            xml,
            n_parallel=n_parallel,
            num_substeps=env.world.params.num_substeps,
            obs_type=env.world.params.obs_type,

        )
        self.env = env
        self.reward = env.reward
        self.custom_reward = custom_reward
        if custom_reward is None:
            self.sense_specs = collect_specs(env.reward).union(extra_specs)
        else:
            self.sense_specs = custom_specs or set()
        self._cached_qpos = None
        self._cached_qvel = None
        self._cached_sense = None

    def reset(self):
        self._cached_qpos = None
        self._cached_qvel = None
        self._cached_sense = None

    def get_relative_frame(self, qpos, qvel, senses=None):
        # TODO: cache

        if senses is None:
            if self._cached_qpos is not None and self._cached_qvel is not None \
                    and qpos.shape == self._cached_qpos.shape \
                    and qvel.shape == self._cached_qvel.shape \
                    and np.allclose(qpos, self._cached_qpos) \
                    and np.allclose(qvel, self._cached_qvel):
                senses = self._cached_sense
            else:
                _, senses = self.mjparallel.forward_dynamics(
                    qpos=qpos,
                    qvel=qvel, ctrl=None, mocap=None, step=False, use_site_xpos=True, use_site_xmat=True)

        gripper_xpos = resolve_site_xpos(self.mjparallel, senses["site_xpos"], "site_stall_mocap_xpos")
        gripper_xmat = resolve_site_xmat(self.mjparallel, senses["site_xmat"], "site_stall_mocap_xmat")

        gripper_xpos = gripper_xpos.reshape(-1, 3)
        gripper_xmat = gripper_xmat.reshape(-1, 3, 3)

        return gripper_xpos, gripper_xmat

    def preprocess_mocap(self, qpos, qvel, mocap):
        mocap_xpos, mocap_euler = np.split(mocap, 2, axis=1)

        if self.env.world.params.obs_type == "relative":
            gripper_xpos, gripper_xmat = self.get_relative_frame(qpos, qvel)

            mocap_xpos *= self.env.world.params.mocap_move_speed
            mocap_euler *= self.env.world.params.mocap_rot_speed

            mocap_xpos = gripper_xpos + np.matmul(gripper_xmat, mocap_xpos.reshape(-1, 3, 1)).reshape(-1, 3)
            mocap_euler = mat2euler_batch(np.matmul(gripper_xmat, euler2mat_batch(mocap_euler)))
        elif self.env.world.params.obs_type == "flatten":
            mocap_xpos = [0.3, 0.0, 0.7] + [0.15, 0.2, 0.1] * mocap_xpos  # TODO: refactor FIXME

        if self.env.world.params.mocap_fix_orientation:
            mocap_euler[:, 0].fill(0)
            mocap_euler[:, 1].fill(0.5 * math.pi)
            mocap_euler[:, 2].fill(0)

        mocap = np.concatenate((mocap_xpos, mocap_euler), axis=1)

        return mocap

    def preprocess_action(self, qpos, qvel, u):
        world = self.env.world
        mocap, ctrl = np.split(copy(u), [world.nmocap * 6], axis=-1)
        assert mocap.shape[-1] == world.nmocap * 6
        assert ctrl.shape[-1] == world.dimu - world.nmocap * 6

        if world.nmocap == 1:
            mocap = self.preprocess_mocap(qpos, qvel, mocap)

            if ctrl.shape[1] == 2:  # gipper
                ctrl[:, 1] = ctrl[:, 0]
                if is_in_rl() or is_in_pi2():
                    ctrl += 0.5
                    ctrl *= 0.04  # FIXME FIXME FIXME

        return mocap, ctrl

    def _extract_robot_pos(self, qpos, qvel):
        if np.prod(qpos.shape) > 0:
            mask = self.env.world._get_robot_indices().astype(np.bool)
            qpos_robot, qvel_robot = \
                self.env.world.control_preprocessor.normalize_observation(
                    qpos[..., mask], qvel[..., mask])
        else:
            qpos_robot = qpos
            qvel_robot = qvel
        return qpos_robot, qvel_robot

    def __call__(self, x, u, t=None, get_obs=False):

        qpos, qvel = np.split(x, [self.env.world.dimq], axis=-1)
        qpos = np.asarray(qpos, dtype=np.float64, order='C')
        qvel = np.asarray(qvel, dtype=np.float64, order='C')
        assert qpos.shape[-1] == self.env.world.dimq
        assert qvel.shape[-1] == self.env.world.dimv
        mocap, ctrl = self.preprocess_action(qpos, qvel, u)

        if np.prod(qpos.shape) > 0 and np.prod(u.shape) > 0:
            mask = self.env.world._get_robot_indices()
            ctrl = self.env.world.control_preprocessor.normalize_action(
                qpos[..., mask], ctrl)
            qpos[..., mask], qvel[..., mask] = \
                self.env.world.control_preprocessor.normalize_observation(
                    qpos[..., mask], qvel[..., mask])

        qpos = np.asarray(qpos, dtype=np.float64, order='C')
        qvel = np.asarray(qvel, dtype=np.float64, order='C')
        ctrl = np.asarray(ctrl, dtype=np.float64, order='C')
        mocap = np.asarray(mocap, dtype=np.float64, order='C')

        sense_specs = self.sense_specs
        if self.env.world.params.obs_type == "relative" or get_obs:
            sense_specs = sense_specs.union({"use_site_xpos", "use_site_xmat"})
            if get_obs:
                sense_specs = sense_specs.union({"use_site_jac"})  # , "use_joint_qpos"})

        xnext, sense = self.mjparallel.forward_dynamics(
            qpos=qpos, qvel=qvel, ctrl=ctrl, mocap=mocap,
            **dict([(x, True) for x in sense_specs]))

        qpos_next = xnext[..., :self.env.world.dimq]
        qvel_next = xnext[..., self.env.world.dimq:]
        sense = dict(sense, qpos=qpos_next, qvel=qvel_next)
        if self.custom_reward is not None:
            rewards = self.custom_reward(xprev=x, xnext=xnext, sense=sense, mjparallel=self.mjparallel, t=t, u=u)
        else:
            rewards = compute_reward(reward=self.reward, sense=sense, mjparallel=self.mjparallel)
        self._cached_qpos = qpos_next
        self._cached_qvel = qvel_next
        self._cached_sense = sense

        if get_obs:
            # compute obs
            obs, diverged = self.get_obs(qpos=qpos_next, qvel=qvel_next, sense=sense)
            return xnext, rewards, sense, obs, diverged
        else:
            return xnext, rewards, sense

    def get_obs(self, qpos, qvel, sense=None):
        if sense is None:
            # Need to run a pass to get senses
            if self.env.world.params.obs_type == "relative":
                sense_specs = {"use_site_xpos", "use_site_xmat", "use_site_jac"}
            else:
                sense_specs = {"use_site_xpos"}
            _, sense = self.mjparallel.forward_dynamics(
                qpos=qpos, qvel=qvel, ctrl=None, mocap=None,
                step=False, **dict([(x, True) for x in sense_specs]))

        if self.env.world.params.obs_type in ["full_state", "flatten", "relative"]:
            qpos_robot, qvel_robot = self._extract_robot_pos(qpos, qvel)

            site_xpos = sense["site_xpos"]
            nsite = self.mjparallel.model.nsite

            if self.env.world.params.obs_type == "full_state":
                obs = list(zip(qpos_robot, qvel_robot, site_xpos))
            elif self.env.world.params.obs_type == "flatten":
                obs = np.concatenate([qpos, qvel, site_xpos], axis=1)
            elif self.env.world.params.obs_type == "relative":
                site_xmat = sense["site_xmat"]
                N = len(qpos)

                stall_mocap_idx = self.mjparallel.site_names.index("stall_mocap")
                gripper_xpos = resolve_site_xpos(self.mjparallel, site_xpos, "site_stall_mocap_xpos").reshape((-1, 3, 1))
                gripper_xmat = resolve_site_xmat(self.mjparallel, site_xmat, "site_stall_mocap_xmat").reshape((-1, 3, 3))

                inv_rot = np.linalg.inv(gripper_xmat).reshape((-1, 3, 3))

                site_xpos = site_xpos.reshape((-1, nsite, 3, 1)) - gripper_xpos[:, None, :, :]
                site_xpos = np.matmul(inv_rot[:, None, :, :], site_xpos).reshape((N, -1))

                # site_xmat = np.matmul(inv_rot[:, None, :, :], site_xmat.reshape(-1, nsite, 3, 3))
                # site_euler = mat2euler_batch(site_xmat.reshape((-1, 3, 3))).reshape((N, -1))

                site_jac_reshaped = sense['site_jac']
                # this is equivalent to reshape(...), # but errors if creating a copy is necessary
                site_jac_reshaped.shape = (N, nsite, 2, 3, self.mjparallel.model.nv)
                site_jacp = site_jac_reshaped[:, :, 0, :, :]


                # velocities

                # N*1*3
                gripper_xvel = np.matmul(site_jacp[:, stall_mocap_idx], qvel.reshape(N, -1, 1)).reshape(N, -1, 3)
                site_xvel = np.matmul(site_jacp, qvel.reshape((N, -1, 1))[:, None, :, :]).reshape(N, -1, 3)
                # site_xvel = np.matmul(site_jacp, sense['qvel'].reshape(1, -1, 1)).reshape(-1,
                #                                                                                       3)  # world coordinates
                site_xvel = np.matmul(inv_rot[:, None, :, :],
                                      (site_xvel - gripper_xvel).reshape(N, -1, 3, 1))  # mocap coordinates

                # TODO: the same for angular velocities
                # gripper state # TODO: change to one scalar when the finger are synchronized
                grip_l = 25. * resolve_joint_qpos(self.mjparallel, qpos, "joint_robot:l_gripper_finger_joint_1_qpos")
                grip_r = 25. * resolve_joint_qpos(self.mjparallel, qpos, "joint_robot:r_gripper_finger_joint_1_qpos")

                site_xpos = site_xpos.reshape((N, -1))
                site_xvel = site_xvel.reshape((N, -1))
                grip_l = grip_l.reshape((N, -1))
                grip_r = grip_r.reshape((N, -1))

                obs = np.concatenate([site_xpos, site_xvel, grip_l, grip_r], axis=1)
            else:
                raise NotImplementedError
            # elif isinstance(self.env.world.params.obs_type, tuple) and self.env.world.params.obs_type[0] == "image":
            #     qpos_robot, qvel_robot = self._extract_robot_pos(x)
            #     if self.renderer is None:
            #         wh = (self.params.obs_type[1], self.params.obs_type[2])
            #         self.renderer = MujocoRenderer(self.local_xml, wh=wh)
            #     obs = (qpos_robot, qvel_robot, np.squeeze(
            #         self.renderer.render(np.expand_dims(x, 0))))
            # elif isinstance(self.params.obs_type, tuple) and self.params.obs_type[0] == "custom":
            #     obs = self.params.obs_type[2](self, x)
            # else:
            #     assert False

            # assert self.observation_space().contains(obs), \
            #     'Value should be in observation space:\nOBS=%s,shape=%s\n\nSPACE=%s' % \
            #     (obs, str([o.shape for o in obs]), self.observation_space())

            if self.env.world.params.divergence_obs_threshold is not None:
                obs_space = self.env.world.observation_space()
                diverged = []
                for obs_i in obs:
                    flatten_obs = obs_space.flatten(obs_i)
                    if len(flatten_obs) > 0 and abs(flatten_obs).max() > self.env.world.params.divergence_obs_threshold:
                        diverged.append(True)
                    else:
                        diverged.append(False)
                return obs, diverged

            return obs, [False] * len(obs)
