import numpy as np
import gc
import os.path as osp
import itertools
from rllab.misc import logger
import collections
from functools import reduce
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class GridBonusEvaluator(object):
    def __init__(self, obs='com', env_spec=None, mesh_density=50):  #it's not great to have policy info here.. but handy for latent
        self.mesh_density = mesh_density
        self.furthest = 0
        self.visitation_all = np.zeros((1, 1), dtype=int)
        self.num_latents = 0  # this will simply not be used if there are no latents (the same for the following 2)
        self.dict_visit = collections.OrderedDict()  # keys: latents (int), values: np.array with number of visitations
        self.visitation_by_lat = np.zeros((1, 1), dtype=int)  # used to plot: matrix with a number for each lat/rep
        # in case I'm gridding all the obs_dim (not just the com) --> for this I should use hashing, ow too high dim
        if env_spec:
            obs_dim = env_spec.observation_space.flat_dim

    def fit_before_process_samples(self, paths):
        """
        NEEDED: Called in process_samples, before processing them. This initializes the hashes based on the current obs.
        """
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
            selectors_name = 'latents' if 'latents' in list(paths[0]['agent_infos'].keys()) else 'selectors'
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

    def predict(self, path):
        """
        NEEDED: Gives the bonus!
        :param path: reward computed path by path
        :return: a 1d array
        """
        freqs = []
        if 'env_infos' in list(path.keys()) and 'com' in list(path['env_infos'].keys()):
            com_x = np.ceil(((np.array(path['env_infos']['com'][:, 0]) + self.furthest) * self.mesh_density)).astype(int)
            com_y = np.ceil(((np.array(path['env_infos']['com'][:, 1]) + self.furthest) * self.mesh_density)).astype(int)
        else:
            com_x = np.ceil(((np.array(path['observations'][:, -2]) + self.furthest) * self.mesh_density)).astype(int)
            com_y = np.ceil(((np.array(path['observations'][:, -3]) + self.furthest) * self.mesh_density)).astype(int)
        coms = list(zip(com_x, com_y))
        lats = [np.nonzero(lat)[0][0] for lat in path['agent_infos']['latents']]
        for i, com in enumerate(coms):
            freqs.append(self.visitation_by_lat[lats[i]][com] / self.visitation_all[com])
        return np.log(freqs)

    def fit_after_process_samples(self, samples_data):
        """
        NEEDED
        """
        pass

    def log_diagnostics(self, paths):
        """
        NEEDED: I will basically plot
        """
        fig, ax = plt.subplots()
        overlap = 0  # keep track of the overlap
        delta = 1./self.mesh_density
        y, x = np.mgrid[-self.furthest:self.furthest+delta:delta, -self.furthest:self.furthest+delta:delta]
        if 'agent_infos' in list(paths[0].keys()) and ('latents' in list(paths[0]['agent_infos'].keys()) or
                                                               'selectors' in list(paths[0]['agent_infos'].keys())):
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
                                     vmax=self.num_latents + 2)  # before 1 (will it affect when no walls?)
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

        visitation_different = np.count_nonzero(self.visitation_all)
        logger.record_tabular('VisitationDifferents', visitation_different)
        logger.record_tabular('VisitationOverlap', overlap)
        logger.record_tabular('VisitationMin', np.min(self.visitation_all))
        logger.record_tabular('VisitationMax', np.max(self.visitation_all))
        total_grid_bonus = np.sum([np.sum(self.predict(path)) for path in paths])
        avg_grid_bonus = np.mean([np.sum(self.predict(path)) for path in paths])
        logger.record_tabular('TotalGridEntropyBonus', total_grid_bonus)
        logger.record_tabular('AvgPathGridEntropyBonus', avg_grid_bonus)
        plt.cla()
        plt.clf()
        plt.close('all')
        # del fig, ax, cmap, cbar, map_plot
        gc.collect()
