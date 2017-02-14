""" Copies trained policies into directory within rllab, for running on EC2
"""
#!/usr/bin/env python

import os
import os.path as osp
import subprocess

SOURCE_DIR = "/home/shhuang/src/rllab-private/data/s3/"
TARGET_DIR = "/home/shhuang/src/rllab-private/sandbox/sandy/adversarial/trained_models_recurrent"
# Experiments to copy over
#EXPERIMENTS = ['trpo/exp027', 'async-rl/exp036', 'deep-q-rl/exp035c']
EXPERIMENTS = ['async-rl/exp037']
PROGRESS_FNAME = 'progress.csv'

def main():
    for exp in EXPERIMENTS:
        parent_dir = osp.join(SOURCE_DIR, osp.dirname(exp))
        exp_dirs = os.listdir(parent_dir)
        exp_dirs = [e for e in exp_dirs \
                    if e.startswith(osp.basename(exp))]
        for exp_dir in exp_dirs:
            params_dirs = [x for x in os.listdir(osp.join(parent_dir, exp_dir))
                           if osp.isdir(osp.join(parent_dir, exp_dir, x))]
            for params_dir in params_dirs:
                params_files = [x for x in os.listdir(osp.join(parent_dir, exp_dir, params_dir)) \
                                if x.startswith('itr') and x.endswith('pkl')]
                # Get the latest parameters
                params_file = sorted(params_files,
                                     key=lambda x: int(x.split('.')[0].split('_')[1]),
                                     reverse=True)[0]
                params_file = osp.join(parent_dir, exp_dir, params_dir, params_file)
                progress_file = osp.join(parent_dir, exp_dir, params_dir, PROGRESS_FNAME)
                target_dir_full = osp.join(TARGET_DIR, osp.dirname(exp), \
                                           exp_dir, params_dir)

                # Make necessary directories
                if not osp.exists(target_dir_full):
                    os.makedirs(target_dir_full)

                # Need the '*' after params_file to copy over any .pkl_01.npy.z files
                cp_out = subprocess.Popen("cp " + params_file + "* " + target_dir_full, \
                                          shell=True, stdout=subprocess.PIPE, \
                                          stderr=subprocess.PIPE)
                print("Copying params file:", params_file, target_dir_full, cp_out.stderr.readlines())
                cp_out = subprocess.Popen("cp " + progress_file + " " + target_dir_full, \
                                          shell=True, stdout=subprocess.PIPE, \
                                          stderr=subprocess.PIPE)
                print("Copying progress file:", progress_file, target_dir_full, cp_out.stderr.readlines())

if __name__ == "__main__":
    main()
