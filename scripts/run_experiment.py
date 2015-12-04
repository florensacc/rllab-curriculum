from __future__ import print_function
from rllab.misc.resolve import load_class
from rllab.misc.console import colorize
from rllab import config
import rllab.misc.logger as logger
import argparse
import sys
import os.path as osp
import datetime
import dateutil.tz
import ast


def instantiate(argvals, cls, *args, **kwargs):
    print(
        colorize(
            'instantiating %s.%s' % (cls.__module__, cls.__name__),
            'green'
        )
    )
    return cls.new_from_args(argvals, *args, **kwargs)


def run_interactive():
    pass


def run_experiment(argv):

    default_log_dir = osp.join(config.PROJECT_PATH, 'data')
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')

    default_exp_name = 'experiment_%s' % timestamp
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='run in interactive mode')
    parser.add_argument('--algo', type=str, metavar='ALGO_PATH',
                        help='module path to the algorithm')
    parser.add_argument('--mdp', type=str, metavar='MDP_PATH',
                        help='module path to the mdp class')
    parser.add_argument('--normalize_mdp', action='store_true',
                        help="Whether to normalize the mdp's actions to take "
                             "value between -1 and 1")
    # These are optional, depending on the algorithm selected
    parser.add_argument('--policy', type=str, metavar='POLICY_PATH',
                        help='module path to the policy')
    parser.add_argument('--vf', type=str, default='no_value_function',
                        help='module path to the value function')
    parser.add_argument('--qf', type=str,
                        help='module path to the Q function')
    parser.add_argument('--es', type=str,
                        help='module path to the exploration strategy')
    parser.add_argument('--more_help', action='store_true',
                        help='whether to show more help depending on the '
                             'classes chosen')
    parser.add_argument('--n_parallel', type=int, default=1,
                        help='Number of parallel workers to perform rollouts.')
    parser.add_argument('--exp_name', type=str, default=default_exp_name,
                        help='Name of the experiment.')
    parser.add_argument('--log_dir', type=str, default=default_log_dir,
                        help='Path to save the log and iteration snapshot.')
    parser.add_argument('--snapshot_mode', type=str, default='all',
                        help='Mode to save the snapshot. Can be either "all" '
                             '(all iterations will be saved), "last" (only '
                             'the last iteration will be saved), or "none" '
                             '(do not save snapshots)')
    parser.add_argument('--tabular_log_file', type=str, default='progress.csv',
                        help='Name of the tabular log file (should end in '
                             '.csv).')
    parser.add_argument('--text_log_file', type=str, default='debug.log',
                        help='Name of the text log file.')
    parser.add_argument('--plot', type=ast.literal_eval, default=False,
                        help='Whether to plot the iteration results')
    parser.add_argument('--seed', type=int,
                        help='Random seed for numpy')

    args = parser.parse_known_args(argv[1:])[0]

    if args.interactive:
        run_interactive()
    else:
        from rllab.sampler import parallel_sampler
        parallel_sampler.init_pool(args.n_parallel)
        if args.plot:
            from rllab.plotter import plotter
            plotter.init_worker()

        from rllab.mdp.base import MDP
        from rllab.vf.base import ValueFunction
        from rllab.policy.base import Policy
        from rllab.qf.base import QFunction
        from rllab.algo.base import Algorithm
        from rllab.es.base import ExplorationStrategy

        if args.seed is not None:
            import numpy as np
            import lasagne
            np.random.seed(args.seed)
            lasagne.random.set_rng(np.random.RandomState(args.seed))
            print(
                colorize(
                    'using seed %s' % (str(args.seed)),
                    'green'
                )
            )

        # Save the arguments which might be useful for later use
        # with open(cache_file, 'w+b') as f:
        #     pickle.dump(sys.argv[1:], f)

        classes = dict()
        classes['mdp'] = load_class(args.mdp, MDP, ["rllab", "mdp"])
        if args.policy:
            classes['policy'] = load_class(
                args.policy, Policy, ["rllab", "policy"])
        if args.qf:
            classes['qf'] = load_class(
                args.qf, QFunction, ["rllab", "qf"])
        if args.es:
            classes['es'] = load_class(
                args.es, ExplorationStrategy, ["rllab", "es"])
        classes['vf'] = load_class(args.vf, ValueFunction, ["rllab", "vf"])
        classes['algo'] = load_class(args.algo, Algorithm, ["rllab", "algo"])

        for cls in classes.values():
            cls.add_args(parser)

        if args.more_help:
            parser.print_help()
            sys.exit(0)

        more_args = parser.parse_known_args(argv[1:])[0]

        instances = dict()
        instances['mdp'] = instantiate(more_args, classes['mdp'])
        if args.normalize_mdp:
            from rllab.mdp.normalized_mdp import normalize
            instances['mdp'] = normalize(instances['mdp'])
        if args.policy:
            instances['policy'] = instantiate(
                more_args, classes['policy'], instances['mdp'])
        if args.qf:
            instances['qf'] = instantiate(
                more_args, classes['qf'], instances['mdp'])
        if args.es:
            instances['es'] = instantiate(
                more_args, classes['es'], instances['mdp'])
        instances['vf'] = instantiate(
            more_args, classes['vf'], instances['mdp'])
        algo = instantiate(more_args, classes['algo'])

        exp_dir = osp.join(args.log_dir, args.exp_name)
        tabular_log_file = osp.join(exp_dir, args.tabular_log_file)
        text_log_file = osp.join(exp_dir, args.text_log_file)
        logger.add_text_output(text_log_file)
        logger.add_tabular_output(tabular_log_file)
        prev_snapshot_dir = logger.get_snapshot_dir()
        prev_mode = logger.get_snapshot_mode()
        logger.set_snapshot_dir(exp_dir)
        logger.set_snapshot_mode(args.snapshot_mode)
        logger.push_prefix("[%s] " % args.exp_name)

        for _ in algo.train(**instances):
            pass
        logger.set_snapshot_mode(prev_mode)
        logger.set_snapshot_dir(prev_snapshot_dir)
        logger.remove_tabular_output(tabular_log_file)
        logger.remove_text_output(text_log_file)
        logger.pop_prefix()

if __name__ == "__main__":
    run_experiment(sys.argv)
