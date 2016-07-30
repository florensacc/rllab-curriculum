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


class reload_policy(object):
    def __init__(self,
                 log_dir, # where to look for the old policy
                 iteration=None,  #if the params.XXX file is specified by its iteration
                 local_root="data/s3",  #root of where to paste the fetched files
                 new_data_dir=None,  # directory on guardar les noves iterations
                 ):

        self.log_dir = log_dir
        self.iteration = iteration
        self.local_root = local_root
        self.new_data_dir = new_data_dir

    def run(self):
        logger.log("Reloading %s, from iteration = %d" % (self.log_dir, self.iteration))

        # download the iteration snapshot
        logger.log("Downloading snapshots and other files...")

        remote_dir = os.path.join(config.AWS_S3_PATH, self.log_dir)

        local_dir = os.path.join(self.local_root, self.log_dir)  #it is NOT the name of new dir to save!
                                                                #should we change it

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
        if self.train_batch_size is None:
            self.train_batch_size = algo.batch_size

        algo.train()

