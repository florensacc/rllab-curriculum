from sandbox.haoran.mddpg.algos.ddpg import DDPG
from sandbox.tuomas.mddpg.algos.vddpg import VDDPG
from rllab.misc import ext
from rllab.misc import logger
from rllab.sampler import parallel_sampler
import tensorflow as tf
import os
import joblib
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('pkl', type=str, default='')
parser.add_argument('--max_path_length', type=int, default=0)
parser.add_argument('--n-paths', type=int, default=1)
parser.add_argument('--n-parallel', type=int, default=1)
parser.add_argument('--logdir', type=str, default='data/tmp')
parser.add_argument('--seed', type=int, default=-1)
args = parser.parse_args()

logger.set_snapshot_dir(args.logdir)

if args.n_parallel > 1:
    # somehow the parallel workers are not able to load the eval_policy, as it
    # thinks that "critic" is already defined in the current session
    raise NotImplementedError
    parallel_sampler.initialize(args.n_parallel)
    if args.seed >= 0:
        parallel_sampler.set_seed(args.seed)
if args.seed >= 0:
    ext.set_seed(args.seed)

with tf.Session() as sess:
    if not os.path.exists(args.pkl):
        print("Cannot find %s"%(args.pkl))
        sys.exit(1)
    else:
        snapshot = joblib.load(args.pkl)

    if "algo" in snapshot:
        algo = snapshot["algo"]
    else:
        # case-by-case re-create the algo
        qf_params = snapshot["qf"].get_param_values()
        pi_params = snapshot["policy"].get_param_values()

        algo = DDPG(
            env=snapshot["env"],
            exploration_strategy=snapshot["es"],
            policy=snapshot["policy"],
            qf=snapshot["qf"],
        )
        algo.qf.set_param_values(qf_params)
        algo.policy.set_param_values(pi_params)
        algo._init_training() # copies params to target networks


    if isinstance(algo, DDPG):
        if args.max_path_length > 0:
            algo.max_path_length = args.max_path_length
        algo.n_eval_samples = algo.max_path_length * args.n_paths
        train_info = dict(
            es_path_returns=[],
            es_path_lengths=[],
        )
        algo._start_worker()
        algo.evaluate(epoch=0, train_info=train_info)
        logger.dump_tabular(with_prefix=False)

    elif isinstance(algo, VDDPG):
        if args.max_path_length > 0:
            algo.max_path_length = args.max_path_length
        algo.n_eval_paths = args.n_paths
        algo._start_worker()
        train_info = dict(
            es_path_returns=[],
            es_path_lengths=[],
        )
        algo._start_worker()
        algo.evaluate(epoch=0, train_info=train_info)
        logger.dump_tabular(with_prefix=False)
    else:
        raise NotImplementedError
