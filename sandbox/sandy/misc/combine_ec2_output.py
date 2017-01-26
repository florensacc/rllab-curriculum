""" Combines all output h5 files from the specified experiment into a single
    h5 output file. Assumes that there are no conflicting keys
"""
#!/usr/bin/env python

import h5py
import os
import os.path as osp

from sandbox.sandy.misc.util import get_time_stamp

BASE_DIR = "/home/shhuang/src/rllab-private/data/s3/adv-rollouts"
EXP_IDX = "exp010"
H5_FNAME = 'fgsm_allvariants.h5'

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

def main():
    cumul_h5 = H5_FNAME.split('.')[0] + '_all_' + get_time_stamp() + '.h5'
    cumul_h5 = osp.join(BASE_DIR, EXP_IDX, cumul_h5)
    print("Output file at", cumul_h5)
    cumul_f = h5py.File(cumul_h5, 'w')

    param_dirs = os.listdir(osp.join(BASE_DIR, EXP_IDX))
    for param_dir in param_dirs:
        if not osp.isdir(osp.join(BASE_DIR, EXP_IDX, param_dir)):
            continue
        partial_h5 = osp.join(BASE_DIR, EXP_IDX, param_dir, H5_FNAME)
        if not osp.isfile(partial_h5):
            print("Skipping dir", param_dir, "because it does not contain", H5_FNAME)
            continue
        print("Adding h5 from dir", param_dir)
        partial_f = h5py.File(partial_h5, 'r')
        copy_over(partial_f, cumul_f)

    cumul_f.close()

if __name__ == "__main__":
    main()
