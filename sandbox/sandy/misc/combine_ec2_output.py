""" Combines all output h5 files from the specified experiment into a single
    h5 output file. Assumes that there are no conflicting keys.
    (This is for adversarial attack experiments, to make analysis of results
    easier.)
"""
#!/usr/bin/env python

import h5py
import itertools
import json
import os
import os.path as osp
import pickle

USE_GPU = True
if USE_GPU:
    import os
    os.environ["THEANO_FLAGS"] = "device=gpu,dnn.enabled=auto,floatX=float32"
    import theano
    print("Theano config:", theano.config.device, theano.config.floatX)

from sandbox.sandy.misc.util import get_time_stamp, to_iterable
from sandbox.sandy.shared.model import load_models

BASE_DIR = "/home/shhuang/src/rllab-private/data/s3/adv-sleeper-rollouts"
EXP_IDX = "exp017b"
SLEEPER = True
H5_FNAME = 'fgsm_allvariants.h5'
PARAMS_FNAME = "params.json"

MODEL_DIR = '/home/shhuang/src/rllab-private/sandbox/sandy/adversarial/trained_models_recurrent/'
AVG_RETURN_KEY = 'avg_return'
H5_BASE_PATH = 'results'
CHOSEN_NORM_EPS_FNAME = ""
#CHOSEN_NORM_EPS_FNAME = "/home/shhuang/src/rllab-private/data/s3/adv-rollouts/exp010/transfer_exp_to_run_20170126_144940_538112.p"

TEST_TRANSFER = False

def copy_over(from_g, to_g):
    for k in from_g:
        is_group = hasattr(from_g[k], 'keys')
        if k not in to_g:
            if is_group:
                to_g.create_group(k)
        if is_group:
            copy_over(from_g[k], to_g[k])
        else:
            if k in to_g:
                if to_g[k][()] != from_g[k][()]:
                    print("\tWARNING: over-writing current key", osp.join(to_g.name, k))
                    del to_g[k]
                    to_g[k] = from_g[k][()]
            else:
                to_g[k] = from_g[k][()]

def get_params_from_experiment(base_dir, exp_idx, params_fname, sleeper=False):
    games = []
    norms = []
    if sleeper:
        k = []
    eps = []
    exp_names = []
    threshold_perf = -1
    threshold_n = -1

    param_dirs = os.listdir(osp.join(base_dir, exp_idx))
    for param_dir in param_dirs:
        if not osp.isdir(osp.join(base_dir, exp_idx, param_dir)):
            continue

        params_file = osp.join(base_dir, exp_idx, param_dir, params_fname)
        if not osp.isfile(params_file):
            print("Skipping dir", param_dir, "because it does not contain", params_fname)
            continue
        params_f = open(params_file).read()
        params = json.loads(params_f)
        params = params['json_args']['algo']
        games += to_iterable(params['games'])
        if sleeper:
            k += [x[0] for x in params['k_init_lambda']]
        norms += to_iterable(params['norms'])
        eps += to_iterable(params['fgsm_eps'])
        exp_names += to_iterable(params['exp_names'])
        threshold_perf = params['threshold_perf']
        threshold_n = params['threshold_n']

    games = list(set(games))
    norms = list(set(norms))
    eps = list(set(eps))
    exp_names = list(set(exp_names))
    if sleeper:
        k = list(set(k))
        return games, norms, eps, exp_names, threshold_perf, threshold_n, k
    return games, norms, eps, exp_names, threshold_perf, threshold_n

def check_missing(cumul_h5, base_dir, exp_idx, params_fname=PARAMS_FNAME):
    # Make sure all norms, exp_names, games, and eps are tested on (i.e., that
    # none of the EC2 runs crashed midway)
    if SLEEPER:
        games, norms, eps, exp_names, threshold_perf, threshold_n, k = \
                get_params_from_experiment(base_dir, exp_idx, params_fname, sleeper=SLEEPER)
    else:
        games, norms, eps, exp_names, threshold_perf, threshold_n = \
                get_params_from_experiment(base_dir, exp_idx, params_fname)

    print("Games:", games)
    print("Norms:", norms)
    print("Eps:", eps)
    print("Experiments:", exp_names)
    print("Threshold Perf:", threshold_perf)
    print("Threshold n:", threshold_n)
    if SLEEPER:
        print("k:", k)

    chosen_norm_eps = None
    if CHOSEN_NORM_EPS_FNAME != "":
        chosen_norm_eps = pickle.load(open(CHOSEN_NORM_EPS_FNAME, 'rb'))

    # Need to make sure A
    cumul_f = h5py.File(cumul_h5, 'r')

    policies = load_models(games, exp_names, MODEL_DIR, \
                           threshold_perf, threshold_n)
    for game in policies:
        game_policies = list(itertools.chain.from_iterable(policies[game].values()))
        #for algo_name in policies[game]:
        #    game_policies += policies[game][algo_name]

        for norm in norms:
            if chosen_norm_eps is not None:
                eps = [x[2] for x in chosen_norm_eps if x[0] == game and x[1] == norm]
                print("Game, norm, eps:", game, norm, eps)

            for e in eps:
                for policy_adv in game_policies:
                    for policy_target in game_policies:
                        if not TEST_TRANSFER and policy_adv != policy_target:
                            continue
                        if SLEEPER:
                            for k_val in k:
                                path = osp.join(H5_BASE_PATH, norm, str(e), policy_adv.model_name, policy_target.model_name, str(k_val), AVG_RETURN_KEY)
                                if path not in cumul_f:
                                    print("WARNING: path", path, "not in file")
                        else:
                            path = osp.join(H5_BASE_PATH, norm, str(e), policy_adv.model_name, policy_target.model_name, AVG_RETURN_KEY)
                            if path not in cumul_f:
                                print("WARNING: path", path, "not in file")
    cumul_f.close()

def combine_ec2_output(h5_fname, base_dir, exp_idx):
    cumul_h5 = h5_fname.split('.')[0] + '_all_' + get_time_stamp() + '.h5'
    cumul_h5 = osp.join(base_dir, exp_idx, cumul_h5)
    print("Output file at", cumul_h5)
    cumul_f = h5py.File(cumul_h5, 'w')

    param_dirs = os.listdir(osp.join(base_dir, exp_idx))
    for param_dir in param_dirs:
        if not osp.isdir(osp.join(base_dir, exp_idx, param_dir)):
            continue
        partial_h5 = osp.join(base_dir, exp_idx, param_dir, h5_fname)
        if not osp.isfile(partial_h5):
            print("Skipping dir", param_dir, "because it does not contain", h5_fname)
            continue
        print("Adding h5 from dir", param_dir)
        partial_f = h5py.File(partial_h5, 'r')
        copy_over(partial_f, cumul_f)

    cumul_f.close()
    return cumul_h5

def main():
    cumul_h5 = combine_ec2_output(H5_FNAME, BASE_DIR, EXP_IDX)
    check_missing(cumul_h5, BASE_DIR, EXP_IDX)

if __name__ == "__main__":
    main()
