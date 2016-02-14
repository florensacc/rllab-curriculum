from __future__ import print_function

from rllab.misc.ext import is_iterable, set_seed
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
import uuid


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

    default_log_dir = "{PROJECT_PATH}/data"
    now = datetime.datetime.now(dateutil.tz.tzlocal())

    # avoid name clashes when running distributed jobs
    rand_id = str(uuid.uuid4())[:5]
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S_%f_%Z')

    default_exp_name = 'experiment_%s_%s' % (timestamp, rand_id)
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='run in interactive mode')
    parser.add_argument('--algo', type=str, metavar='ALGO_PATH',
                        help='module path to the algorithm')
    parser.add_argument('--mdp', type=str, metavar='MDP_PATH',
                        help='module path to the mdp class')
    parser.add_argument('--normalize_mdp', type=ast.literal_eval,
                        default=False,
                        help="Whether to normalize the mdp's actions to take "
                             "value between -1 and 1")
    parser.add_argument('--random_mdp', type=ast.literal_eval,
                        default=False,
                        help="Whether to reinit the mdp from random physical "
                             "model every episode")
    parser.add_argument('--action_delay', type=int,
                        default=0,
                        help="Time steps delayed injected into MDP")
    parser.add_argument('--obs_noise', type=float,
                        default=0,
                        help="Guassian noise added to obs")
    # These are optional, depending on the algorithm selected
    parser.add_argument('--policy', type=str, metavar='POLICY_PATH',
                        help='module path to the policy')
    parser.add_argument('--baseline', type=str,
                        help='module path to the baseline')
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
                        help='Name of the tabular log file (in csv).')
    parser.add_argument('--text_log_file', type=str, default='debug.log',
                        help='Name of the text log file (in pure text).')
    parser.add_argument('--params_log_file', type=str, default='params.json',
                        help='Name of the parameter log file (in json).')
    parser.add_argument('--plot', type=ast.literal_eval, default=False,
                        help='Whether to plot the iteration results')
    parser.add_argument('--seed', type=int,
                        help='Random seed for numpy')

    args = parser.parse_known_args(argv[1:])[0]

    if args.interactive:
        run_interactive()
    else:
        from rllab.sampler import parallel_sampler
        parallel_sampler.config_parallel_sampler(args.n_parallel, args.seed)
        if args.plot:
            from rllab.plotter import plotter
            plotter.init_worker()

        from rllab.mdp.base import MDP
        from rllab.baseline.base import Baseline
        from rllab.policy.base import Policy
        from rllab.algo.base import Algorithm

        if args.seed is not None:
            set_seed(args.seed)

        # Save the arguments which might be useful for later use
        # with open(cache_file, 'w+b') as f:
        #     pickle.dump(sys.argv[1:], f)

        classes = dict()
        classes['mdp'] = load_class(args.mdp, MDP, ["rllab", "mdp"])
        if args.policy:
            classes['policy'] = load_class(
                args.policy, Policy, ["rllab", "policy"])
        if args.baseline:
            classes['baseline'] = load_class(
                args.baseline, Baseline, ["rllab", "baseline"])
        classes['algo'] = load_class(args.algo, Algorithm, ["rllab", "algo"])

        for cls in classes.values():
            cls.add_args(parser)

        if args.more_help:
            parser.print_help()
            sys.exit(0)

        more_args = parser.parse_args(argv[1:])

        instances = dict()
        if args.random_mdp:
            from rllab.mdp.identification_mdp import IdentificationMDP
            instances['mdp'] = IdentificationMDP(classes['mdp'], more_args)
        else:
            instances['mdp'] = instantiate(more_args, classes['mdp'])
        if args.normalize_mdp:
            from rllab.mdp.normalized_mdp import normalize
            instances['mdp'] = normalize(instances['mdp'])
        if args.action_delay != 0:
            from rllab.mdp.noisy_mdp import DelayedActionMDP
            instances['mdp'] = DelayedActionMDP(
                instances['mdp'], args.action_delay)
        if args.obs_noise != 0:
            from rllab.mdp.noisy_mdp import NoisyObservationMDP
            instances['mdp'] = NoisyObservationMDP(
                instances['mdp'], args.obs_noise)
        if args.policy:
            instances['policy'] = instantiate(
                more_args, classes['policy'], instances['mdp'])
        if args.baseline:
            instances['baseline'] = instantiate(
                more_args, classes['baseline'], instances['mdp'])
        algo = instantiate(more_args, classes['algo'])

        log_dir = args.log_dir.format(PROJECT_PATH=config.PROJECT_PATH)
        exp_dir = osp.join(log_dir, args.exp_name)
        tabular_log_file = osp.join(exp_dir, args.tabular_log_file)
        text_log_file = osp.join(exp_dir, args.text_log_file)
        params_log_file = osp.join(exp_dir, args.params_log_file)

        logger.log_parameters(params_log_file, more_args, classes)
        logger.add_text_output(text_log_file)
        logger.add_tabular_output(tabular_log_file)
        prev_snapshot_dir = logger.get_snapshot_dir()
        prev_mode = logger.get_snapshot_mode()
        logger.set_snapshot_dir(exp_dir)
        logger.set_snapshot_mode(args.snapshot_mode)
        logger.push_prefix("[%s] " % args.exp_name)

        maybe_iter = algo.train(**instances)
        if is_iterable(maybe_iter):
            for _ in maybe_iter:
                pass

        logger.set_snapshot_mode(prev_mode)
        logger.set_snapshot_dir(prev_snapshot_dir)
        logger.remove_tabular_output(tabular_log_file)
        logger.remove_text_output(text_log_file)
        logger.pop_prefix()

if __name__ == "__main__":
    run_experiment(sys.argv)
