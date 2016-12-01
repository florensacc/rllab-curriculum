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
    def __init__(self, obs='com', env_spec=None, mesh_density=50, visitation_bonus=1.0, snn_H_bonus=0, survival_bonus=0):
        self.mesh_density = mesh_density
        self.furthest = 0
        self.visitation_all = np.zeros((1, 1), dtype=int)
        self.num_latents = 0  # this will simply not be used if there are no latents (the same for the following 2)
        self.dict_visit = collections.OrderedDict()  # keys: latents (int), values: np.array with number of visitations
        self.visitation_by_lat = np.zeros((1, 1), dtype=int)  # used to plot: matrix with a number for each lat/rep
        self.visitation_bonus = visitation_bonus
        self.snn_H_bonus = snn_H_bonus
        self.survival_bonus = survival_bonus
        # in case I'm gridding all the obs_dim (not just the com) --> for this I should use hashing, ow too high dim
        if env_spec:
            obs_dim = env_spec.observation_space.flat_dim

    def fit_before_process_samples(self, paths):
        """
        NEEDED: Called in process_samples, before processing them. This initializes the hashes based on the current obs.
        """
        if 'env_infos' in paths[0].keys() and 'full_path' in paths[0]['env_infos'].keys():
            paths = [tensor_utils.flatten_first_axis_tensor_dict(path['env_infos']['full_path']) for path in paths]

        if 'env_infos' in list(paths[0].keys()) and 'com' in list(paths[0]['env_infos'].keys()):
            x_max = np.ceil(np.max(np.abs(np.concatenate([path["env_infos"]['com'][:, 0] for path in paths]))))
            y_max = np.ceil(np.max(np.abs(np.concatenate([path["env_infos"]['com'][:, 1] for path in paths]))))
        else:
            x_max = np.ceil(np.max(np.abs(np.concatenate([path["observations"][:, -2] for path in paths]))))
            y_max = np.ceil(np.max(np.abs(np.concatenate([path["observations"][:, -3] for path in paths]))))
        self.furthest = max(x_max, y_max)
        print('THE FUTHEST IT WENT COMPONENT-WISE IS: x_max={}, y_max={}'.format(x_max, y_max))
        if 'agent_infos' in list(paths[0].keys()) and (('latents' in list(paths[0]['agent_infos'].keys())
                                                        and np.size(paths[0]['agent_infos']['latents'])) or
                                                           ('selectors' in list(paths[0]['agent_infos'].keys())
                                                            and np.size(paths[0]['agent_infos']['selectors']))):
            selectors_name = 'selectors' if 'selectors' in list(paths[0]['agent_infos'].keys()) else 'latents'
            self.num_latents = np.size(paths[0]["agent_infos"][selectors_name][0])
            # set all the labels for the latents and initialize the entries of dict_visit
            for i in range(self.num_latents):  # use integer to define the latents
                self.dict_visit[i] = np.zeros((2 * self.furthest * self.mesh_density + 1, 2 * self.furthest * self.mesh_density + 1))
            for path in paths:
                # before this was [1][0] !! Idk why not it changed, but [0][0] should be the correct one!
                lats = [np.nonzero(lat)[0][0] for lat in path['agent_infos'][selectors_name]]  # list of all lats by idx
                if 'env_infos' in list(paths[0].keys()) and 'com' in list(paths[0]['env_infos'].keys()):
                    com_x = np.ceil(((np.array(path['env_infos']['com'][:, 0]) + self.furthest) * self.mesh_density)).astype(int)
                    com_y = np.ceil(((np.array(path['env_infos']['com'][:, 1]) + self.furthest) * self.mesh_density)).astype(int)
                else:
                    com_x = np.ceil(((np.array(path['observations'][:, -2]) + self.furthest) * self.mesh_density)).astype(int)
                    com_y = np.ceil(((np.array(path['observations'][:, -3]) + self.furthest) * self.mesh_density)).astype(int)
                coms = list(zip(com_x, com_y))
                for i, com in enumerate(coms):
                    self.dict_visit[lats[i]][com] += 1
            self.visitation_all = reduce(np.add, [visit for visit in self.dict_visit.values()])

        else:  # If I don't have latents
            self.visitation_all = np.zeros((2 * self.furthest * self.mesh_density + 1, 2 * self.furthest * self.mesh_density + 1))
            for path in paths:
                if 'env_infos' in list(paths[0].keys()) and 'com' in list(paths[0]['env_infos'].keys()):
                    com_x = np.ceil(((np.array(path['env_infos']['com'][:, 0]) + self.furthest) * self.mesh_density)).astype(int)
                    com_y = np.ceil(((np.array(path['env_infos']['com'][:, 1]) + self.furthest) * self.mesh_density)).astype(int)
                else:
                    com_x = np.ceil(((np.array(path['observations'][:, -2]) + self.furthest) * self.mesh_density)).astype(int)
                    com_y = np.ceil(((np.array(path['observations'][:, -3]) + self.furthest) * self.mesh_density)).astype(int)
                coms = list(zip(com_x, com_y))
                for com in coms:
                    self.visitation_all[com] += 1

    def predict_count(self, path):
        if 'env_infos' in path.keys() and 'full_path' in path['env_infos'].keys():
            path = tensor_utils.flatten_first_axis_tensor_dict(path['env_infos']['full_path'])

        counts = []
        if 'env_infos' in list(path.keys()) and 'com' in list(path['env_infos'].keys()):
            com_x = np.ceil(((np.array(path['env_infos']['com'][:, 0]) + self.furthest) * self.mesh_density)).astype(int)
            com_y = np.ceil(((np.array(path['env_infos']['com'][:, 1]) + self.furthest) * self.mesh_density)).astype(int)
        else:
            com_x = np.ceil(((np.array(path['observations'][:, -2]) + self.furthest) * self.mesh_density)).astype(int)
            com_y = np.ceil(((np.array(path['observations'][:, -3]) + self.furthest) * self.mesh_density)).astype(int)
        coms = list(zip(com_x, com_y))
        for com in coms:
            counts.append(self.visitation_all[com])
        return 1. / np.maximum(1., np.sqrt(counts))

    def predict_entropy(self, path):
        if 'env_infos' in path.keys() and 'full_path' in path['env_infos'].keys():
            path = tensor_utils.flatten_first_axis_tensor_dict(path['env_infos']['full_path'])

        if 'env_infos' in list(path.keys()) and 'com' in list(path['env_infos'].keys()):
            com_x = np.ceil(((np.array(path['env_infos']['com'][:, 0]) + self.furthest) * self.mesh_density)).astype(int)
            com_y = np.ceil(((np.array(path['env_infos']['com'][:, 1]) + self.furthest) * self.mesh_density)).astype(int)
        else:
            com_x = np.ceil(((np.array(path['observations'][:, -2]) + self.furthest) * self.mesh_density)).astype(int)
            com_y = np.ceil(((np.array(path['observations'][:, -3]) + self.furthest) * self.mesh_density)).astype(int)
        coms = list(zip(com_x, com_y))
        freqs = []
        lats = [np.nonzero(lat)[0][0] for lat in path['agent_infos']['latents']]
        for i, com in enumerate(coms):
            freqs.append(self.dict_visit[lats[i]][com] / self.visitation_all[com])
        return np.log(freqs)

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

        if self.snn_H_bonus:  # I need the if because the snn bonus is only available when there are latents
            bonus = self.snn_H_bonus * self.predict_entropy(expanded_path) + \
                    self.visitation_bonus * self.predict_count(expanded_path)
        else:
            bonus = self.visitation_bonus * self.predict_count(expanded_path)
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
        delta = 1./self.mesh_density
        y, x = np.mgrid[-self.furthest:self.furthest+delta:delta, -self.furthest:self.furthest+delta:delta]
        if 'agent_infos' in list(paths[0].keys()) and (('latents' in list(paths[0]['agent_infos'].keys())
                                                        and np.size(paths[0]['agent_infos']['latents'])) or
                                                           ('selectors' in list(paths[0]['agent_infos'].keys())
                                                            and np.size(paths[0]['agent_infos']['selectors']))):
            # fix the colors for each latent
            num_colors = self.num_latents + 2  # +2 for the 0 and Repetitions NOT COUNTING THE WALLS
            # create a matrix with entries corresponding to the latent that was there (or other if several/wall/nothing)
            self.visitation_by_lat = np.zeros((2 * self.furthest * self.mesh_density + 1, 2 * self.furthest * self.mesh_density + 1))
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
            overlap = np.sum(np.where(self.visitation_all > 1, self.visitation_all, 0))  # sum of all visitations larger than 1
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

        # if self.survival_bonus:
        avg_survival_bonus = np.mean([len(path['rewards']) for path in paths])
        logger.record_tabular('AvgPath_SurviBonus', avg_grid_count_bonus)

        avg_grid_bonus = np.mean([np.sum(self.predict(path)) for path in paths])
        logger.record_tabular('AvgPathGridBonus', avg_grid_bonus)
