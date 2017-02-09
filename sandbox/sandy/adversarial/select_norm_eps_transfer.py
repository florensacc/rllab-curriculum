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
SELECTION_TYPE = 'lower_than'  # Options: "in_range", "lower_than"
EXP_IDX_TO_NAME = {'exp027': 'trpo_exp027',
                   'exp036': 'async-rl_exp036',
                   'exp035c': 'deep-q-rl_exp035c'}

def get_returns_by_game_exp_norm(h5_file, norm_key=NORM_KEY, eps_key=EPS_KEY, \
                                 policy_adv_key=POLICY_ADV_KEY, \
                                 avg_return_key=AVG_RETURN_KEY, \
                                 exp_idx_to_name=EXP_IDX_TO_NAME):
    param_names = get_param_names(h5_file)
    norm_idx = param_names.index(norm_key)
    eps_idx = param_names.index(eps_key)
    policy_adv_idx = param_names.index(policy_adv_key)
    paths = get_all_result_paths(h5_file, avg_return_key)

    f = h5py.File(h5_file, 'r')
    # Separate out all results by game and norm
    avg_returns = {}  # keys: (game, norm)
    for path in paths:
        path_info = path.split('/')[2:]
        norm = path_info[norm_idx]
        game = path_info[policy_adv_idx].split('_')[-1]
        exp_name = exp_idx_to_name[path_info[policy_adv_idx].split('_')[0]]
        if game == 'invaders':
            game = 'space-invaders'
        elif game == 'command':
            game = 'chopper-command'

        key = game + '/' + exp_name + '/' + norm
        if key not in avg_returns:
            avg_returns[key] = []
        avg_returns[key].append((path, float(path_info[eps_idx]), f[path][avg_return_key][()]))
    f.close()
    return avg_returns

def select_interesting_norm_eps(h5_file, range_ratio=0.7, lower_than_ratios=[0.25, 0.5, 0.75, 0.9], \
                                split_by_exp=False):
    # "Interesting" is defined as where the agent's performance starts to decrease,
    # but it hasn't completely broken yet
    # range_ratio: percentage of the range (centered between max and
    #              min performance) that we consider interesting.
    #              Only used if SELECTION_TYPE == 'in_range'
    # lower_than_ratio: percentage of max performance that agent
    #                   performance should be lower than, for selected epsilon,
    #                   Only used if SELECTION_TYPE == "lower_than"

    avg_returns = get_returns_by_game_exp_norm(h5_file)
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
    avg_return_keys = sorted(avg_returns.keys())
    for key in avg_return_keys:
        eps_avg_returns = avg_returns[key]
        max_avg_return = np.max([x[1] for x in eps_avg_returns])
        min_avg_return = np.min([x[1] for x in eps_avg_returns])
        range_size = (max_avg_return - min_avg_return)

        if SELECTION_TYPE == "in_range":
            # Figure out which epsilons fall within interesting range of performance
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
        elif SELECTION_TYPE == "lower_than":
            # Pick first epsilon that's below 50% of max performance, and the
            # one after that
            chosen_epsilons[key] = []
            lower_than_ratios = sorted(lower_than_ratios)
            print(key)
            print("Avg return max and min:", max_avg_return, min_avg_return)
            for ratio in lower_than_ratios:
                range_max = max_avg_return - ratio * range_size
                interesting_eps = [x for x in eps_avg_returns \
                                   if x[1] <= range_max]
                interesting_eps = sorted(interesting_eps, key=lambda x: x[0])
                if interesting_eps[0][0] not in chosen_epsilons[key]:
                    chosen_epsilons[key].append(interesting_eps[0][0])
                print("\tRange max:", range_max)
                #print("\t", interesting_eps)
            print('\t', chosen_epsilons[key])

    print(chosen_epsilons)
    exp_to_run = []
    for key in chosen_epsilons:
        if split_by_exp:
            game, exp_name, norm = key.split('/')
        else:
            game, norm = key.split('/')
        game = game.split('-')[0]
        for eps in chosen_epsilons[key]:
            if split_by_exp:
                exp_to_run.append((exp_name, game, norm, eps))
            else:
                exp_to_run.append((game, norm, eps))

    exp_to_run = sorted(exp_to_run, key=lambda x: (x[0]+x[1],x[2]))

    output_fname = osp.join(osp.dirname(h5_file), 'transfer_exp_to_run_' + get_time_stamp() + '.p')
    print("Saving in", output_fname)
    pickle.dump(exp_to_run, open(output_fname, "wb"))
    # pickle.load(open(output_fname, 'rb'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('returns_h5', type=str)
    args = parser.parse_args()

    #select_interesting_norm_eps(args.returns_h5)
    select_interesting_norm_eps(args.returns_h5, lower_than_ratios=[0.5], split_by_exp=True)

if __name__ == "__main__":
    main()
