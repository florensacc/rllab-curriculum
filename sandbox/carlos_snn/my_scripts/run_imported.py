"""
this class simply wraps all methods
"""
import theano
import theano.tensor as T
import numpy as np
import sys
import argparse
import time
import os
import joblib

from rllab import config
from rllab.sampler import parallel_sampler
from rllab.misc import ext
import rllab.misc.logger as logger

from sandbox.carlos_snn.my_scripts.importer import load_problem

sys.path.append('.')


class Warm_starter(object):
    def __init__(self,
                 log_dir,
                 iteration=None,
                 n_parallel=4,
                 local_root="data/s3",
                 new_data_dir=None,
                 ):

        self.log_dir = log_dir
        self.iteration = iteration  # this is important as it will load the pickle file 'itr_XX.pkl'
        self.local_root = local_root
        self.n_parallel = n_parallel
        self.new_data_dir = new_data_dir

    def run(self):
        if self.iteration:
            logger.log("Analyzing %s, iteration = %d" % (self.log_dir, self.iteration))
        else:
            logger.log("Analyzing %s, last iteration" % self.log_dir)

        # download the iteration snapshot
        logger.log("Downloading snapshots and other files...")
        remote_dir = os.path.join(config.AWS_S3_PATH, self.log_dir)

        # set where to download it
        if self.new_data_dir:  #if we want to change its title.
            local_dir = os.path.join(self.local_root, self.new_data_dir)
        else:
            local_dir = os.path.join(self.local_root, self.log_dir)

        if not os.path.isdir(local_dir):
            os.system("mkdir -p %s" % (local_dir))

        if self.iteration:
            pkl_file_name="itr_{}.pkl".format(self.iteration)
        else:
            pkl_file_name="params.pkl"

        file_names = [
            pkl_file_name,
            "params.json",
            "progress.csv",
        ]

        for file_name in file_names:
            remote_file = os.path.join(remote_dir, file_name)
            command = """
                aws s3 cp {remote_file} {local_dir}/.
            """.format(remote_file=remote_file, local_dir=local_dir)
            os.system(command)

        # load problem ----------------------------------------------  # problem : not being able to change env!!
        problem = load_problem(local_dir, self.iteration)
        algo = problem["algo"]

        algo.init_opt()
        algo.train()

