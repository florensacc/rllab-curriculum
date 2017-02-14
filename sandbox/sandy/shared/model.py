#!/usr/bin/env python

import os, os.path as osp
from sandbox.sandy.shared.config import EXP_IDX_TO_NAME
from sandbox.sandy.shared.model_load import load_model
from sandbox.sandy.shared.experiment import get_experiments
from sandbox.sandy.shared.config import SCORE_KEY

class TrainedModel(object):
    #(algo, env, score, params_file.split('/')[-2])
    def __init__(self, model_dir, exp, score_window=10, load=True):
        # Loads most recent model from the directory
        # (e.g., params.pkl or itr_##.pkl from rllab)
        self.model_dir = osp.join(exp.full_path, model_dir)
        self.exp = exp
        _, self.model_name = osp.split(self.model_dir)
        self.score_window = score_window
        self.model_file, self.score = get_latest_model(self.model_dir, self.exp, score_window)
        if load:
            self.algo, self.env = load_model(self.model_file)

def model_dir_to_exp_name(model_dir):
    return EXP_IDX_TO_NAME[model_dir.split('_')[0]]

def model_file_to_itr(f):
    return int(f.split('.')[0].split('_')[1])

def get_latest_model(model_dir, exp, score_window, scores_file="progress.csv"):
    model_files = [x for x in os.listdir(model_dir) \
                   if x.startswith('itr') and x.endswith('pkl')]
    # Get the latest parameters
    model_file = sorted(model_files, key=lambda x: model_file_to_itr(x), \
                        reverse=True)[0]
    itr = model_file_to_itr(model_file)

    # Calculate average score starting from saved iteration (itr) and
    # going back score_window iterations
    with open(osp.join(model_dir, scores_file), 'r') as progress_f:
        lines = progress_f.readlines()
        header = lines[0].strip().split(',')
        score_idx = header.index(SCORE_KEY[exp.algo_name])
        scores = [float(l.split(',')[score_idx]) \
                  for l in lines[max(1,itr-score_window):itr+1]]
        score = sum(scores) / float(len(scores))

    return osp.join(model_dir, model_file), score

def get_all_models(exp):
    # Gets model from each directory in the specified experiment
    # exp - instantiation of Experiment
    model_dirs = [osp.join(exp.full_path,x)
                  for x in os.listdir(exp.full_path)
                  if osp.isdir(osp.join(exp.full_path,x))]
    models = [TrainedModel(model_dir, exp) for model_dir in model_dirs]
    return models

def load_models(games, exp_names, base_dir, threshold=0, \
                num_threshold=5, score_window=10, load_model=True):
    # experiments should be in the format "<algo-name>_<exp_index>"
    # threshold - discard all policies with a score less than threshold*best_score,
    #             for each game and training algorithm pair
    # num_threshold - discard all but the top num_threshold policies, even if
    #                 they meet the threshold
    threshold = max(min(threshold, 1), 0)

    policies = {}  # key, top level: game name
                   # key, second level: algorithm name
                   # value: list of (algo, env) pairs - trained policies for that game
    experiments = get_experiments(games, exp_names, base_dir)
    for exp in experiments:
        exp_models = get_all_models(exp)
        if exp.game not in policies:
            policies[exp.game] = {}
        if exp.algo_name not in policies[exp.game]:
            policies[exp.game][exp.algo_name] = []

        policies[exp.game][exp.algo_name] += exp_models

    # Discard all policies that are not almost as good as the best one, or not
    # in the top num_threshold scores
    best_policies = {}
    for game in policies:
        best_policies[game] = {}
        for algo_name in policies[game]:
            all_policies = sorted(policies[game][algo_name], \
                                  key=lambda x: x.score, reverse=True)
            best_score = all_policies[0].score
            best_policies[game][algo_name] = [x for x in all_policies \
                                              if x.score >= best_score*threshold][:num_threshold]
    return best_policies
