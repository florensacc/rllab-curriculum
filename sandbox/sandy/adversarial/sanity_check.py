#!/usr/bin/env python

import h5py

from sandbox.sandy.adversarial.io_util import get_all_result_paths

# Make sure EC2 runs are reproducible
WITH_TRANSFER_H5 = "/home/shhuang/src/rllab-private/data/s3/adv-rollouts/exp011/fgsm_allvariants_all_20170126_212420_640404.h5"
WITHOUT_TRANSFER_H5 = "/home/shhuang/src/rllab-private/data/s3/adv-rollouts/exp010/fgsm_allvariants_all_20170126_131149_355616.h5"
AVG_RETURN_KEY = 'avg_return'

def check_identical(check_from_h5, check_in_h5):
    result_paths = get_all_result_paths(check_from_h5, AVG_RETURN_KEY)
    check_from_f = h5py.File(check_from_h5, 'r')
    check_in_f = h5py.File(check_in_h5, 'r')
    for p in result_paths:
        if p in check_in_f:
            check_in_val = check_in_f[p][AVG_RETURN_KEY][()]
            check_from_val = check_from_f[p][AVG_RETURN_KEY][()]
            if check_in_val != check_from_val:
                print("Mismatch at path", p, ":", check_from_val, "vs.", check_in_val)

def main():
    check_identical(WITHOUT_TRANSFER_H5, WITH_TRANSFER_H5)

if __name__ == "__main__":
    main()
