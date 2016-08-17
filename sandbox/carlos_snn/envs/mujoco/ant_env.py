from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs

import matplotlib as mpl
# from matplotlib.figure import Figure
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

mpl.use('Agg')
import matplotlib.pyplot as plt
import os.path as osp

import gc

class AntEnv(MujocoEnv, Serializable):
    FILE = 'ant.xml'

    @autoargs.arg('ctrl_cost_coeff', type=float,
                  help='cost coefficient for controls')
    def __init__(self,
                 ctrl_cost_coeff=1e-2,
                 rew_speed=False,  # if True the dot product is taken with the speed instead of the position
                 rew_dir=None,  # (x,y,z) -> Rew=dot product of the CoM SPEED with this dir. Otherwise, DIST to 0
                 *args, **kwargs):
        self.ctrl_cost_coeff = ctrl_cost_coeff
        self.reward_dir = rew_dir
        self.rew_speed = rew_speed

        super(AntEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)  # locals()????

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
        if self.rew_speed:
            direction_com = self.get_body_comvel('torso')
        else:
            direction_com = self.get_body_com('torso')
        if self.reward_dir:
            direction = np.array(self.reward_dir, dtype=float)/np.linalg.norm(self.reward_dir)
            print 'my direction of reward:', direction
            print 'my comvel: ', direction_com
            forward_reward = np.dot(direction, direction_com)
            print "the dot prod, ", forward_reward
        else:
            forward_reward = np.linalg.norm(direction_com[0:-1])  # instead of comvel[0] (does this give jumping reward??)
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(np.square(action / scaling))
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),  # what is this??
        survive_reward = 0.05  # this is not in swimmer neither!!
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self._state
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self.get_current_obs()
        return Step(ob, float(reward), done)

    @overrides
    def log_diagnostics(self, paths):
        # plt.close('all')
        # print "printing tracker diff ENTER log_diag of env:"
        # self.tr.print_diff()
        progs = [
            np.linalg.norm(path["observations"][-1][-3:-1] - path["observations"][0][-3:-1])
            for path in paths
            ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))

        # now we will grid the space and check how much of it the policy is covering
        furthest = np.ceil(np.abs(np.max(np.concatenate([path["observations"][:, -3:-1] for path in paths]))))
        print 'THE FUTHEST IT WENT COMPONENT-WISE IS', furthest
        furthest = max(furthest, 5)
        c_grid = int(furthest * 10 * 2)

        if 'agent_infos' in paths[0].keys() and 'latents' in paths[0]['agent_infos'].keys():
            dict_visit = {}
            # keep track of the overlap
            overlap = 0
            for path in paths:
                lat = str(path['agent_infos']['latents'][0])
                if lat not in dict_visit.keys():
                    dict_visit[lat] = np.zeros((c_grid + 1, c_grid + 1))
                com_x = np.clip(np.ceil(((np.array(path['observations'][:, -3]) + furthest) * 10)).astype(int), 0,
                                c_grid)
                com_y = np.clip(np.ceil(((np.array(path['observations'][:, -2]) + furthest) * 10)).astype(int), 0,
                                c_grid)
                coms = zip(com_x, com_y)
                for com in coms:
                    dict_visit[lat][com] += 1
            num_latents = len(dict_visit.keys())
            num_colors = num_latents + 2  # +2 for the 0 and Repetitions
            cmap = plt.get_cmap('nipy_spectral', num_colors)
            visitation_by_lat = np.zeros((c_grid + 1, c_grid + 1))
            for i, visit in enumerate(dict_visit.itervalues()):
                lat_visit = np.where(visit == 0, visit, i + 1)  # transform the map into 0 or i+1
                visitation_by_lat += lat_visit
                overlap += np.sum(np.where(visitation_by_lat > lat_visit))  # add the overlaps of this latent
                visitation_by_lat = np.where(visitation_by_lat <= i + 1, visitation_by_lat,
                                             num_colors - 1)  # mark overlaps
            x = np.arange(c_grid + 1) / 10. - furthest
            y = np.arange(c_grid + 1) / 10. - furthest

            fig = plt.figure(0)
            # fig = Figure()
            # canvas = FigureCanvas(fig)
            # fig.clf()
            ax = fig.add_subplot(111)
            map_plot = ax.pcolormesh(x, y, visitation_by_lat, cmap=cmap, vmin=0.1, vmax=num_latents + 1)
            color_len = (num_colors - 1.) / num_colors
            ticks = np.arange(color_len / 2., num_colors - 1, color_len)
            cbar = fig.colorbar(map_plot, ticks=ticks)
            latent_tick_labels = ['latent: ' + l for l in dict_visit.keys()]
            cbar.ax.set_yticklabels(['No visitation'] + latent_tick_labels + ['Repetitions'])  # horizontal colorbar

            # still log the total visitation and the overlap
            visitation = reduce(np.add, [visit for visit in dict_visit.itervalues()])
        else:
            visitation = np.zeros((c_grid + 1, c_grid + 1))
            for path in paths:
                com_x = np.clip(np.ceil(((np.array(path['observations'][:, -3]) + furthest) * 10)).astype(int), 0,
                                c_grid)
                com_y = np.clip(np.ceil(((np.array(path['observations'][:, -2]) + furthest) * 10)).astype(int), 0,
                                c_grid)
                coms = zip(com_x, com_y)
                for com in coms:
                    visitation[com] += 1
            x = np.arange(c_grid + 1) / 10. - furthest
            y = np.arange(c_grid + 1) / 10. - furthest

            fig = plt.figure(0)
            ax = fig.add_subplot(111)
            map_plot = ax.pcolormesh(x, y, visitation, vmax=10)

            overlap = np.sum(np.where(visitation > 1, visitation, 0))  # sum of all visitations larger than 1
        ax.set_xlim([x[0], x[-1]])
        ax.set_ylim([y[0], y[-1]])

        total_visitation = np.count_nonzero(visitation)
        logger.record_tabular('VisitationTotal', total_visitation)
        logger.record_tabular('VisitationOverlap', overlap)

        log_dir = logger.get_snapshot_dir()
        exp_name = log_dir.split('/')[-1]
        plt.title(exp_name)

        plt.savefig(osp.join(log_dir, 'visitation.png'))

        plt.cla()
        plt.clf()
        plt.close('all')
        del fig, ax, cmap, cbar, map_plot
        gc.collect()

        # print "printing the tracker after log_diagnostics:"
        # self.tr.print_diff()

