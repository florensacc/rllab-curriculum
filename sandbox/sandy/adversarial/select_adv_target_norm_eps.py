#!/usr/bin/python

""" Selects which adversarial policy + target policy + epsilon + norm combos
    to visualize for the arXiv website, http://rll.berkeley.edu/adversarial
"""

import argparse
import h5py
import numpy as np
import os.path as osp
import pickle

from sandbox.sandy.adversarial.io_util import get_param_names, get_all_result_paths
from sandbox.sandy.adversarial.select_norm_eps_transfer import get_returns_by_game_exp_norm, EXP_IDX_TO_NAME
from sandbox.sandy.shared.model_load import load_models
from sandbox.sandy.misc.util import get_time_stamp

SELECTION_TYPE = 'lower_than'  # Options: "in_range", "lower_than"
GAMES = ['chopper', 'pong', 'seaquest', 'space']
EXPERIMENTS = ['trpo_exp027', 'async-rl_exp036', 'deep-q-rl_exp035c']  # Format: algo-name_exp-index
MODEL_DIR = '/home/shhuang/src/rllab-private/sandbox/sandy/adversarial/trained_models/'
BATCH_SIZE = None

def get_name_top_policy(best_policies, game, exp_name, idx=0):
    # idx = 0 means best, = 1 means second-best, etc.
    return best_policies[game.split('-')[0]][exp_name.split('_')[0]][idx][-1]

def select_adv_target_norm_eps(h5_file, range_ratio=0.7, lower_than_ratios=[0.25, 0.5, 0.75, 0.9], \
                               split_by_exp=False):
    # Selects interesting combinations where the agent's performance starts to decrease,
    # but it hasn't completely broken yet
    # range_ratio: percentage of the range (centered between max and
    #              min performance) that we consider interesting.
    #              Only used if SELECTION_TYPE == 'in_range'
    # lower_than_ratio: percentage of max performance that agent
    #                   performance should be lower than, for selected epsilon,
    #                   Only used if SELECTION_TYPE == "lower_than"

    avg_returns = get_returns_by_game_exp_norm(h5_file)

    # Only pay attention to top two models for each game + training algo combo
    best_policies = load_models(GAMES, EXPERIMENTS, MODEL_DIR, \
                                threshold=0, num_threshold=2, load_model=False)

    # key: game, exp_name
    max_min_scores = {}

    potential_exps = {}
    for k in avg_returns:
        game, exp_name_adv, norm = k.split('/')
        for path, eps, ret in avg_returns[k]:
            policy_adv, policy_target = path.split('/')[-2:]
            exp_name_target = EXP_IDX_TO_NAME[policy_target.split('_')[0]]
            key = '/'.join([game, policy_adv, policy_target, norm])

            max_min_key = '/'.join([game, exp_name_target])
            if max_min_key not in max_min_scores:
                max_min_scores[max_min_key] = [float('-inf'), float('inf')]
            max_min_scores[max_min_key] = [max(ret, max_min_scores[max_min_key][0]), \
                                           min(ret, max_min_scores[max_min_key][1])]

            # White-box, black-box policy, and black-box algo
            if (policy_target == get_name_top_policy(best_policies, game, exp_name_adv, idx=0) and \
                policy_adv == get_name_top_policy(best_policies, game, exp_name_adv, idx=0)) or \
               (policy_target == get_name_top_policy(best_policies, game, exp_name_adv, idx=0) and \
                policy_adv == get_name_top_policy(best_policies, game, exp_name_adv, idx=1)) or \
               (policy_target == get_name_top_policy(best_policies, game, exp_name_target, idx=0) and \
                policy_adv == get_name_top_policy(best_policies, game, exp_name_adv, idx=0)):
                   if key not in potential_exps:
                       potential_exps[key] = []
                   potential_exps[key].append((eps, ret))

    # Select epsilon for each key in potential_exps
    exp_to_run = []
    for k,eps_rets in potential_exps.items():
        game, policy_adv, policy_target, norm = k.split('/')
        exp_name_target = EXP_IDX_TO_NAME[policy_target.split('_')[0]]
        eps_rets = sorted(eps_rets, key=lambda x: x[0])
        if eps_rets[0][0] == 0:
            eps_rets = eps_rets[1:]
        threshold = np.mean(max_min_scores['/'.join([game,exp_name_target])])
        valid_eps_rets = [x[0] for x in eps_rets if x[1] < threshold]
        if len(valid_eps_rets) > 0:
            exp_to_run.append((policy_adv, policy_target, norm, min(valid_eps_rets)))
        else:
            exp_to_run.append((policy_adv, policy_target, norm, eps_rets[-1][0]))
            print("DID NOT MEET THRESHOLD:", threshold, exp_to_run[-1])
            print(eps_rets[-1])

    exp_to_run = sorted(exp_to_run, key=lambda x: (x[0]+x[1],x[2]))
    output_fname = osp.join(osp.dirname(h5_file), 'adv_target_norm_eps_' + get_time_stamp() + '.p')
    print("Saving in", output_fname)
    pickle.dump(exp_to_run, open(output_fname, "wb"))
    # pickle.load(open(output_fname, 'rb'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('returns_h5', type=str)
    args = parser.parse_args()

    #select_interesting_norm_eps(args.returns_h5)
    select_adv_target_norm_eps(args.returns_h5, lower_than_ratios=[0.5], split_by_exp=True)

if __name__ == "__main__":
    main()
