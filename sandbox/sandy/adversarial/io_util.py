import h5py
import joblib
import numpy as np
import os

from sandbox.sandy.misc.util import create_dir_if_needed, get_time_stamp

def init_all_output_file(output_dir, adv_name, algo_param_names, batch_size, \
                         fname=None):
    create_dir_if_needed(output_dir)
    if fname is None:
        fname = adv_name + '_allvariants.h5'
    output_h5 = os.path.join(output_dir, fname)
    
    f = h5py.File(output_h5, 'w')
    f['adv_type'] = adv_name
    f['algo_param_names'] = ';'.join(algo_param_names)
    f['batch_size'] = batch_size
    f.create_group("results")
    f.close()

    return output_h5

def save_performance_to_all(output_h5, avg_return, adv_params, n_paths):
    f = h5py.File(output_h5, 'r+')
    algo_param_names = f['algo_param_names'][()].split(';')
    g = f
    for p in algo_param_names:
        if str(adv_params[p]) not in g:
            g.create_group(str(adv_params[p]))
        g = g[str(adv_params[p])]
    g['avg_return'] = avg_return
    g['n_paths'] = n_paths
    f.close()

def init_output_file(output_dir, prefix, adv_name, adv_params, fname=None):
    create_dir_if_needed(output_dir)
    if fname is None:
        fname = prefix + '_' + get_time_stamp() + '.h5'
    output_h5 = os.path.join(output_dir, fname)

    f = h5py.File(output_h5, 'w')
    f.create_group('rollouts')
    f['adv_type'] = adv_name
    f.create_group('adv_params')
    for k,v in adv_params.items():
        f['adv_params'][k] = v
    f.close()
    return output_h5

def save_rollout_step(output_h5, eta, unscaled_eta, obs, adv_obs):
    output_f = h5py.File(output_h5, 'r+')
    idx = len(output_f['rollouts'])
    g = output_f['rollouts'].create_group(str(idx))
    g['change'] = eta
    g['change_unscaled'] = unscaled_eta
    g['orig_input'] = obs
    g['adv_input'] = adv_obs
    output_f.close()

def save_performance(output_h5, avg_return, n_paths):
    f = h5py.File(output_h5, 'r+')
    f['avg_return'] = avg_return
    f['n_paths'] = n_paths
    f.close()

def save_video_file(output_h5, video_file):
    f = h5py.File(output_h5, 'r+')
    f['rollouts_video'] = video_file
    f.close()
