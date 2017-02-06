#!/usr/bin/env python

""" One-off script for figure out which game-norm-eps combinations have not
been run on EC2 yet
"""
BASE_DIR = "/home/shhuang/src/rllab-private/data/s3/adv-rollouts"
EXP_IDX = "exp010"
PARAMS_FNAME = "params.json"

import pickle
import os.path as osp

from sandbox.sandy.misc.combine_ec2_output import get_params_from_experiment
from sandbox.sandy.misc.util import get_time_stamp

CHOSEN_NORM_EPS_FNAME = "/home/shhuang/src/rllab-private/data/s3/adv-rollouts/exp010/transfer_exp_to_run_20170126_144940_538112.p"

def main():
    chosen_norm_eps = pickle.load(open(CHOSEN_NORM_EPS_FNAME, 'rb'))
    games, norms, eps, _ = get_params_from_experiment(BASE_DIR, EXP_IDX, PARAMS_FNAME)

    leftover_norm_eps = set()
    for g in games:
        for n in norms:
            for e in eps:
                leftover_norm_eps.add((g,n,e))
    leftover_norm_eps = leftover_norm_eps - set(chosen_norm_eps)
    leftover_norm_eps = list(leftover_norm_eps)
    leftover_norm_eps = sorted(leftover_norm_eps)

    leftover_norm_eps_triple = []
    triple = None  # actually has max of four, not three
    for game_norm_eps in leftover_norm_eps:
        if triple is None:
            triple = [game_norm_eps]
        elif len(triple) == 4 or not (triple[0][0] == game_norm_eps[0] and triple[0][1] == game_norm_eps[1]):
            triple_eps = [x[2] for x in triple]
            leftover_norm_eps_triple.append((triple[0][0], triple[0][1], triple_eps))
            triple = [game_norm_eps]
        else:
            triple.append(game_norm_eps)
    if triple is not None:
        triple_eps = [x[2] for x in triple]
        leftover_norm_eps_triple.append((triple[0][0], triple[0][1], triple_eps))

    output_fname = osp.join(osp.dirname(CHOSEN_NORM_EPS_FNAME), 'leftover_transfer_exp_to_run_' + get_time_stamp() + '.p')
    print("Saving in", output_fname)
    pickle.dump(leftover_norm_eps_triple, open(output_fname, "wb"))

if __name__ == "__main__":
    main()
