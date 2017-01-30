import numpy as np
import gc
import os.path as osp
import itertools
from rllab.misc import logger
from rllab.misc import tensor_utils
import collections
from functools import reduce
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt


class GridBonusEvaluator(object):
    def __init__(self,
                 obs='com',
                 env_spec=None,
                 mesh_density=50,
                 visitation_bonus=0,
                 snn_H_bonus=0,
                 virtual_reset=False,  # the paths are split by latents and every switch gets the robot to 0 (xy,ori)
                 switch_lat_every=0,
                 survival_bonus=0,
                 dist_from_reset_bonus=0,
                 start_bonus_after=0):
        self.mesh_density = mesh_density
        self.furthest = 0
        self.visitation_all = np.zeros((1, 1), dtype=int)
        self.num_latents = 0  # this will simply not be used if there are no latents (the same for the following 2)
        self.dict_visit = collections.OrderedDict()  # keys: latents (int), values: np.array with number of visitations
        self.visitation_by_lat = np.zeros((1, 1), dtype=int)  # used to plot: matrix with a number for each lat/rep
        self.visitation_bonus = visitation_bonus
        self.snn_H_bonus = snn_H_bonus
        self.virtual_reset = virtual_reset
        self.switch_lat_every = switch_lat_every
        self.survival_bonus = survival_bonus
        self.dist_from_reset_bonus = dist_from_reset_bonus
        self.start_bonus_after = start_bonus_after
        # in case I'm gridding all the obs_dim (not just the com) --> for this I should use hashing, or too high dim
        if env_spec:
            obs_dim = env_spec.observation_space.flat_dim

    def fit_before_process_samples(self, paths):
        """
        NEEDED: Called in process_samples, before processing them. This initializes the hashes based on the current obs.
        """
        if 'env_infos' in paths[0].keys() and 'full_path' in paths[0]['env_infos'].keys():
            paths = [tensor_utils.flatten_first_axis_tensor_dict(path['env_infos']['full_path']) for path in paths]

        if 'env_infos' in list(paths[0].keys()) and 'com' in list(paths[0]['env_infos'].keys()):
            coms_xy = [np.array(path['env_infos']['com'][:, 0:2]) for path in paths]  # no z coord
        else:
            coms_xy = [np.array(path['observations'][:, -3:-1])[:, [1, 0]] for path in paths]

        if self.virtual_reset:  # change the com according to switch_lat_every or resets
            for k, com_xy in enumerate(coms_xy):
                i = self.start_bonus_after
                while i < len(com_xy):
                    start = i
                    ori = paths[k]['env_infos']['ori'][i - self.start_bonus_after]
                    c = np.cos(ori)
                    s = np.sin(ori)
                    R = np.matrix('{} {}; {} {}'.format(c[0], -s[0], s[0], c[0]))
                    while i < len(com_xy) and i - start < self.switch_lat_every - self.start_bonus_after:
                        i += 1
                    com_xy[start:i] = np.dot(R, com_xy[start:i].T).T
                    xy = com_xy[start]
                    com_xy[start:i] -= xy
                    while i < len(com_xy) and i - start < self.switch_lat_every:  # skip some! compare to above
                        i += 1

        self.furthest = np.ceil(np.max(np.abs(np.concatenate(coms_xy))))
        # now translate and scale the coms!
        coms = [np.ceil((com_xy + self.furthest) * self.mesh_density).astype(int) for com_xy in coms_xy]

        if 'agent_infos' in list(paths[0].keys()) and (('latents' in list(paths[0]['agent_infos'].keys())
                                                        and np.size(paths[0]['agent_infos']['latents'])) or
                                                           ('selectors' in list(paths[0]['agent_infos'].keys())
                                                            and np.size(paths[0]['agent_infos']['selectors']))):
            selectors_name = 'selectors' if 'selectors' in list(paths[0]['agent_infos'].keys()) else 'latents'
            self.num_latents = np.size(paths[0]["agent_infos"][selectors_name][0])
            # set all the labels for the latents and initialize the entries of dict_visit
            size_grid = int(2 * self.furthest * self.mesh_density + 1)
            for i in range(self.num_latents):  # use integer to define the latents
                self.dict_visit[i] = np.zeros((size_grid, size_grid))
            lats = [[np.nonzero(lat)[0][0] for lat in path['agent_infos'][selectors_name]]
                    for path in paths]  # list of all lats by idx
            for k, com in enumerate(coms):  # this iterates through paths
                start = 0
                for i, xy in enumerate(com):
                    if i - start == self.switch_lat_every:
                        start = i
                    if i - start < self.start_bonus_after:
                        pass
                    else:
                        self.dict_visit[lats[k][i]][tuple(xy)] += 1

            self.visitation_all = reduce(np.add, [visit for visit in self.dict_visit.values()])

        else:  # If I don't have latents. I also assume no virtual reset and no start_bonus_after!!
            self.visitation_all = np.zeros(
                (2 * self.furthest * self.mesh_density + 1, 2 * self.furthest * self.mesh_density + 1))
            for com in np.concatenate(coms):
                self.visitation_all[tuple(com)] += 1

    def predict_count(self, path):
        if 'env_infos' in path.keys() and 'full_path' in path['env_infos'].keys():
            path = tensor_utils.flatten_first_axis_tensor_dict(path['env_infos']['full_path'])

        if 'env_infos' in list(path.keys()) and 'com' in list(path['env_infos'].keys()):
            com_xy = np.array(path['env_infos']['com'][:, 0:2])
        else:
            com_xy = np.array(path['observations'][:, -3:-1])[:, [1, 0]]

        if self.virtual_reset:  # change the com according to switch_lat_every or resets
            i = self.start_bonus_after
            while i < len(com_xy):
                start = i
                ori = path['env_infos']['ori'][i - self.start_bonus_after]
                c, s = np.cos(ori), np.sin(ori)
                R = np.matrix('{} {}; {} {}'.format(c[0], -s[0], s[0], c[0]))
                while i < len(com_xy) and i - start < self.switch_lat_every - self.start_bonus_after:
                    i += 1
                com_xy[start:i] = np.dot(R, com_xy[start:i].T).T
                xy = com_xy[start]
                com_xy[start:i] -= xy
                while i < len(com_xy) and i - start < self.switch_lat_every:  # skip some! compare to above
                    i += 1

        # now translate and scale the coms!
        coms = np.ceil((com_xy + self.furthest) * self.mesh_density).astype(int)

        counts = []
        start = 0
        for i, com in enumerate(coms):
            if i - start == self.switch_lat_every:
                start = i
            if i - start < self.start_bonus_after:
                counts.append(np.inf)  # this is the way of zeroing out the reward for the first steps
            else:
                counts.append(self.visitation_all[tuple(com)])

        return 1. / np.maximum(1., np.sqrt(counts))

    def predict_entropy(self, path):
        if 'env_infos' in path.keys() and 'full_path' in path['env_infos'].keys():
            path = tensor_utils.flatten_first_axis_tensor_dict(path['env_infos']['full_path'])

        if 'env_infos' in list(path.keys()) and 'com' in list(path['env_infos'].keys()):
            com_xy = np.array(path['env_infos']['com'][:, 0:2])
        else:
            com_xy = np.array(path['observations'][:, -3:-1])[:, [1, 0]]

        if self.virtual_reset:  # change the com according to switch_lat_every or resets
            i = self.start_bonus_after
            while i < len(com_xy):
                start = i
                ori = path['env_infos']['ori'][i - self.start_bonus_after]
                c, s = np.cos(ori), np.sin(ori)
                R = np.matrix('{} {}; {} {}'.format(c[0], -s[0], s[0], c[0]))
                while i < len(com_xy) and i - start < self.switch_lat_every - self.start_bonus_after:
                    i += 1
                com_xy[start:i] = np.dot(R, com_xy[start:i].T).T
                xy = com_xy[start]
                com_xy[start:i] -= xy
                while i < len(com_xy) and i - start < self.switch_lat_every:  # skip some! compare to above
                    i += 1

        # now translate and scale the coms!
        coms = np.ceil((com_xy + self.furthest) * self.mesh_density).astype(int)

        freqs = []
        lats = [np.nonzero(lat)[0][0] for lat in path['agent_infos']['latents']]

        start = 0
        for i, com in enumerate(coms):
            if i - start == self.switch_lat_every:
                start = i
            if i - start < self.start_bonus_after:
                freqs.append(
                    1.)  # this is tricky because it will be higher than the other rewards!! (negatives) -> at least bonus for staying alife until the transition
            else:
                freqs.append(self.dict_visit[lats[i]][tuple(com)] / self.visitation_all[tuple(com)])
        return np.log(freqs)

    def predict_dist_from_reset(self, path):
        if 'env_infos' in path.keys() and 'full_path' in path['env_infos'].keys():
            path = tensor_utils.flatten_first_axis_tensor_dict(path['env_infos']['full_path'])

        if 'env_infos' in list(path.keys()) and 'com' in list(path['env_infos'].keys()):
            com_xy = np.array(path['env_infos']['com'][:, 0:2])
        else:
            com_xy = np.array(path['observations'][:, -3:-1])[:, [1, 0]]

        if self.virtual_reset:  # change the com according to switch_lat_every or resets
            i = self.start_bonus_after
            while i < len(com_xy):
                start = i
                ori = path['env_infos']['ori'][i - self.start_bonus_after]
                c, s = np.cos(ori), np.sin(ori)
                R = np.matrix('{} {}; {} {}'.format(c[0], -s[0], s[0], c[0]))
                while i < len(com_xy) and i - start < self.switch_lat_every - self.start_bonus_after:
                    i += 1
                com_xy[start:i] = np.dot(R, com_xy[start:i].T).T
                xy = com_xy[start]
                com_xy[start:i] -= xy
                while i < len(com_xy) and i - start < self.switch_lat_every:  # skip some! compare to above
                    i += 1

        # now translate and scale the coms!
        coms = np.ceil((com_xy + self.furthest) * self.mesh_density).astype(int)

        dists_from_reset = []

        start = 0
        for i, com in enumerate(coms):
            if i - start == self.switch_lat_every:
                start = i
            if i - start < self.start_bonus_after:
                dists_from_reset.append(
                    0.)  # this is tricky because it will be higher than the other rewards!! (negatives) -> at least bonus for staying alife until the transition
            else:
                dists_from_reset.append(np.linalg.norm(com - coms[start + self.start_bonus_after]))
        return np.array(dists_from_reset)

    def predict(self, path):
        """
        NEEDED: Gives the bonus!
        :param path: reward computed path by path
        :return: a 1d array
        """
        if 'env_infos' in path.keys() and 'full_path' in path['env_infos'].keys():
            expanded_path = tensor_utils.flatten_first_axis_tensor_dict(path['env_infos']['full_path'])
        else:  # when it comes from log_diagnostics it's already expanded (or if it was never aggregated)
            expanded_path = path

        bonus = self.visitation_bonus * self.predict_count(expanded_path) + \
                self.dist_from_reset_bonus * self.predict_dist_from_reset(expanded_path)
        if self.snn_H_bonus:  # I need the if because the snn bonus is only available when there are latents
            bonus += self.snn_H_bonus * self.predict_entropy(expanded_path)

        total_bonus = bonus + self.survival_bonus * np.ones_like(bonus)

        if 'env_infos' in path.keys() and 'full_path' in path['env_infos'].keys():
            aggregated_bonus = []
            full_path_rewards = path['env_infos']['full_path']['rewards']
            total_steps = 0
            for sub_rewards in full_path_rewards:
                aggregated_bonus.append(np.sum(total_bonus[total_steps:total_steps + len(sub_rewards)]))
                total_steps += len(sub_rewards)
            total_bonus = aggregated_bonus

        return np.array(total_bonus)

    def fit_after_process_samples(self, samples_data):
        """
        NEEDED
        """
        pass

    def log_diagnostics(self, paths):
        """
        NEEDED: I will basically plot
        """
        if 'env_infos' in paths[0].keys() and 'full_path' in paths[0]['env_infos'].keys():
            paths = [tensor_utils.flatten_first_axis_tensor_dict(path['env_infos']['full_path']) for path in paths]

        fig, ax = plt.subplots()
        overlap = 0  # keep track of the overlap
        delta = 1. / self.mesh_density
        y, x = np.mgrid[-self.furthest:self.furthest + delta:delta, -self.furthest:self.furthest + delta:delta]
        if 'agent_infos' in list(paths[0].keys()) and (('latents' in list(paths[0]['agent_infos'].keys())
                                                        and np.size(paths[0]['agent_infos']['latents'])) or
                                                           ('selectors' in list(paths[0]['agent_infos'].keys())
                                                            and np.size(paths[0]['agent_infos']['selectors']))):
            # fix the colors for each latent
            num_colors = self.num_latents + 2  # +2 for the 0 and Repetitions NOT COUNTING THE WALLS
            # create a matrix with entries corresponding to the latent that was there (or other if several/wall/nothing)
            size_grid = int(2 * self.furthest * self.mesh_density + 1)
            self.visitation_by_lat = np.zeros(
                (size_grid, size_grid))
            for i, visit in self.dict_visit.items():
                lat_visit = np.where(visit == 0, visit, i + 1)  # transform the map into 0 or i+1
                self.visitation_by_lat += lat_visit
                overlap += np.sum(np.where(self.visitation_by_lat > lat_visit))  # add the overlaps of this latent
                self.visitation_by_lat = np.where(self.visitation_by_lat <= i + 1, self.visitation_by_lat,
                                                  num_colors - 1)  # mark overlaps
            cmap = plt.get_cmap('nipy_spectral', num_colors)
            map_plot = ax.pcolormesh(x, y, self.visitation_by_lat, cmap=cmap, vmin=0.1,
                                     vmax=self.num_latents + 1)  # before 1 (will it affect when no walls?)
            color_len = (num_colors - 1.) / num_colors
            ticks = np.arange(color_len / 2., num_colors - 1, color_len)
            cbar = fig.colorbar(map_plot, ticks=ticks)
            latent_tick_labels = ['latent: ' + str(i) for i in list(self.dict_visit.keys())]
            cbar.ax.set_yticklabels(
                ['No visitation'] + latent_tick_labels + ['Repetitions'])  # horizontal colorbar
        else:
            plt.pcolormesh(x, y, self.visitation_all, vmax=self.mesh_density)
            overlap = np.sum(
                np.where(self.visitation_all > 1, self.visitation_all, 0))  # sum of all visitations larger than 1
        ax.set_xlim([x[0][0], x[0][-1]])
        ax.set_ylim([y[0][0], y[-1][0]])

        log_dir = logger.get_snapshot_dir()
        exp_name = log_dir.split('/')[-1] if log_dir else '?'
        ax.set_title('visitation_Bonus: ' + exp_name)

        plt.savefig(osp.join(log_dir, 'visitation_Gbonus.png'))  # this saves the current figure, here f
        plt.close()

        plt.cla()
        plt.clf()
        plt.close('all')
        # del fig, ax, cmap, cbar, map_plot
        gc.collect()

        visitation_different = np.count_nonzero(self.visitation_all)
        logger.record_tabular('VisitationDifferents', visitation_different)
        logger.record_tabular('VisitationOverlap', overlap)
        logger.record_tabular('VisitationMin', np.min(self.visitation_all))
        logger.record_tabular('VisitationMax', np.max(self.visitation_all))

        if self.snn_H_bonus:
            avg_grid_entropy_bonus = np.mean([np.sum(self.predict_entropy(path)) for path in paths])
            logger.record_tabular('AvgPath_Grid_EntropyBonus', avg_grid_entropy_bonus)

        # if self.visitation_bonus:
        avg_grid_count_bonus = np.mean([np.sum(self.predict_count(path)) for path in paths])
        logger.record_tabular('AvgPath_Grid_CountBonus', avg_grid_count_bonus)

        # if self.visitation_bonus:
        avg_grid_dist_bonus = np.mean([np.sum(self.predict_dist_from_reset(path)) for path in paths])
        logger.record_tabular('AvgPath_Grid_DistBonus', avg_grid_dist_bonus)

        # if self.survival_bonus:
        avg_survival_bonus = np.mean([len(path['rewards']) for path in paths])
        logger.record_tabular('AvgPath_SurviBonus', avg_survival_bonus)

        avg_grid_bonus = np.mean([np.sum(self.predict(path)) for path in paths])
        logger.record_tabular('AvgPathGridBonus', avg_grid_bonus)
