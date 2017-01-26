#!/usr/bin/python

import argparse
import h5py
import numpy as np
import os.path as osp
import pickle

from sandbox.sandy.adversarial.io_util import get_param_names, get_all_result_paths
from sandbox.sandy.misc.util import get_time_stamp

AVG_RETURN_KEY = "avg_return"
NORM_KEY = "norm"
EPS_KEY = "fgsm_eps"
POLICY_ADV_KEY = "policy_adv"

def select_interesting_norm_eps(h5_file, range_ratio=0.7):
    # "Interesting" is defined as where the agent's performance starts to decrease,
    # but it hasn't completely broken yet
    # range_ratio defines the percentage of the range (centered between max and
    # min performance) that we consider interesting.
    param_names = get_param_names(h5_file)
    norm_idx = param_names.index(NORM_KEY)
    eps_idx = param_names.index(EPS_KEY)
    policy_adv_idx = param_names.index(POLICY_ADV_KEY)
    paths = get_all_result_paths(h5_file, AVG_RETURN_KEY)

    f = h5py.File(h5_file, 'r')
    # Separate out all results by game and norm
    avg_returns = {}  # keys: (game, norm)
    for path in paths:
        path_info = path.split('/')[2:]
        norm = path_info[norm_idx]
        game = path_info[policy_adv_idx].split('_')[-1]
        if game == 'invaders':
            game = 'space-invaders'
        elif game == 'command':
            game = 'chopper-command'

        key = game + '_' + norm
        if key not in avg_returns:
            avg_returns[key] = []
        avg_returns[key].append((path, float(path_info[eps_idx]), f[path][AVG_RETURN_KEY][()]))

    # Average the scores for each epsilon
    for key in avg_returns:
        all_paths = avg_returns[key]
        avg_return_per_eps = {}
        for p in all_paths:
            if p[1] not in avg_return_per_eps:
                avg_return_per_eps[p[1]] = []
            avg_return_per_eps[p[1]].append(p[2])
        avg_returns[key] = []
        for eps in avg_return_per_eps:
            avg_return = sum(avg_return_per_eps[eps]) / float(len(avg_return_per_eps[eps]))
            avg_returns[key].append((eps, avg_return))

    chosen_epsilons = {}
    for key in avg_returns:
        eps_avg_returns = avg_returns[key]
        max_avg_return = np.max([x[1] for x in eps_avg_returns])
        min_avg_return = np.min([x[1] for x in eps_avg_returns])

        # Figure out which epsilons fall within interesting range of performance
        range_size = (max_avg_return - min_avg_return)
        range_max = max_avg_return - range_size * (0.5 - range_ratio/2.0)
        range_min = min_avg_return + range_size * (0.5 - range_ratio/2.0)
        interesting_eps = [x for x in eps_avg_returns if x[1] <= range_max and x[1] >= range_min]
        interesting_eps = sorted(interesting_eps, key=lambda x: x[0])
        print(key)
        print("Avg return max and min:", max_avg_return, min_avg_return)
        print("Range max and min:", range_max, range_min)
        print("\t", len(interesting_eps))
        print("\t", interesting_eps)
        chosen_epsilons[key] = [x[0] for x in interesting_eps]

    f.close()

    print(chosen_epsilons)
    exp_to_run = []
    for key in chosen_epsilons:
        game, norm = key.split('_')
        game = game.split('-')[0]
        for eps in chosen_epsilons[key]:
            exp_to_run.append((game, norm, eps))

    exp_to_run = sorted(exp_to_run, key=lambda x: (x[0]+x[1],x[2]))

    output_fname = osp.join(osp.dirname(h5_file), 'transfer_exp_to_run_' + get_time_stamp() + '.p')
    pickle.dump(exp_to_run, open(output_fname, "wb"))
    # pickle.load(open(output_fname, 'rb'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('returns_h5', type=str)
    args = parser.parse_args()

    select_interesting_norm_eps(args.returns_h5)

if __name__ == "__main__":
    main()
