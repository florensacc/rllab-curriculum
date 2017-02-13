#!/usr/bin/env python

import os, os.path as osp

def get_experiments(games, exp_names, base_dir):
    # exp_names should be in the format "<algo-name>_<exp_index>"
    experiments = []
    for game in games:
        for exp_name in exp_names:
            algo_name, exp_index = exp_name.split('_')
            exp_path = osp.join(base_dir, algo_name, exp_index+'-'+game)
            experiments.append(Experiment(exp_path))
    return experiments

class Experiment(object):
    def __init__(self, exp_path):
        # exp_path - should be an absolute path; contains one or more directories,
        #            one for each of the variants run for this experiment
        # This assumes the experiment path is in a specific format,
        # e.g. the last part should be "<algo-name>/<exp-index>-<game>"
        self.full_path = exp_path
        base_dir, exp_dir = osp.split(exp_path)
        self.exp_index, self.game = exp_dir.split('-')
        self.base_dir, self.algo_name = osp.split(base_dir)
