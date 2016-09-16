from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.misc.overrides import overrides
import numpy as np
from timeit import default_timer as timer


class LinearFeatureBaseline_par(LinearFeatureBaseline):
    def __init__(self, **kwargs):
        super(LinearFeatureBaseline_par, self).__init__(**kwargs)
        self._l_max = 0.

    def _features_traj(self, obs, l):
        o = np.clip(obs, -10, 10)
        return np.concatenate([o, o ** 2, self._al[:l], self._al2[:l], self._al3[:l], self._ones[:l]], axis=1)

    def _features_matrix(self, obs, start_indeces):
        lengths = [s2 - s1 for s1,s2 in zip(start_indeces[:-1],start_indeces[1:])]
        # (the last entry in start indeces is where the next traj would start)
        l_max = max(lengths)
        if l_max > self._l_max:
            self._l_max = l_max
            self._al = np.arange(l_max).reshape(-1, 1) / 100.0
            self._al2 = self._al ** 2
            self._al3 = self._al ** 3
            self._ones = np.ones((l_max,1))
        
        return np.concatenate([self._features_traj(obs[s1:s2],l) 
            for s1,s2,l in zip(start_indeces[:-1], start_indeces[1:], lengths)], axis=0)


    def _fit_bare_input(self, obs, returns, start_indeces):
        # Rather than being fed a list of dicts ("paths") as in the serial case, and extracting
        # the relevant data, this expects to be fed bare arrays of only the relevant data.
        # This minimizes the overhead of sharing data between processes--the three inputs
        # here are the only samples data that needs to be shared among processes in
        # the entire optimization algorithm.
        featmat = self._features_matrix(obs, start_indeces)
        self._coeffs = np.linalg.lstsq(
            featmat.T.dot(featmat) + self._reg_coeff * np.identity(featmat.shape[1]),
            featmat.T.dot(returns)
        )[0]  


    def share_samples(self, samples_data, par_data, baseline_shareds):
        """
        Assumes that a fixed number of time-steps is sampled by each process.
        (This code would have to change for variable-length sample reporting.)
        """
        rank = par_data['rank']
        n_proc = par_data['n_proc']
        batch_size = par_data['batch_size']

        obs_s = baseline_shareds['observations'][rank]
        ret_s = baseline_shareds['returns'][rank]
        starts_s = baseline_shareds['start_indeces'][rank]
        n_start_s = baseline_shareds['num_start_indeces'][rank]

        obs_s[:] = np.reshape(samples_data['observations'], (-1,1))
        ret_s[:] = np.concatenate([path['returns'] for path in samples_data['paths']])

        n_start_w = len(samples_data['paths'])
        starts_w = [0] * n_start_w
        for i,path in enumerate(samples_data['paths'][:-1]):
            starts_w[i+1] = starts_w[i] + len(path['rewards'])
        starts_s[:n_start_w] = starts_w
        n_start_s.value = n_start_w    

        # Just in the last proc, append what would be the start of next trajectory
        # (baseline fitting code will be expecting this).
        if rank == n_proc - 1:
            starts_s[n_start_w] = batch_size
            n_start_s.value += 1


    def fit_par(self, par_data, baseline_shareds):
        """
        To be called by only one process.
        """
        size_obs = par_data['size_obs']
        batch_size = par_data['batch_size']

        obs_s = baseline_shareds['observations'] # (list: one for each proc)
        ret_s = baseline_shareds['returns']
        starts_s = baseline_shareds['start_indeces']
        n_starts_s = baseline_shareds['num_start_indeces']
        coeffs_s = baseline_shareds['coeffs'] # (only one exists)

        stamp_start = timer()

        obs_all = np.concatenate([o[:] for o in obs_s]).reshape([-1, size_obs])
        ret_all = np.concatenate([r[:] for r in ret_s])
        starts_all = np.concatenate([ np.frombuffer(s, dtype='int32')[:n.value] + p * batch_size 
            for p,(s,n) in enumerate(zip(starts_s, n_starts_s)) ]) # (frombuffer() for broadcasted addition)
        stamp_copy = timer()

        self._fit_bare_input(obs_all, ret_all, starts_all) # (updates baseline._coeffs)
        stamp_fit = timer()
        coeffs_s[:] = self._coeffs

        time_copy = stamp_copy - stamp_start
        time_fit = stamp_fit - stamp_copy

        times = {'copy': time_copy,
                 'fit': time_fit}

        return times


    def read_par(self, baseline_shareds):
        self._coeffs[:] = baseline_shareds['coeffs']

