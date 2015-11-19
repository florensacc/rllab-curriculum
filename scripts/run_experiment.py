import argparse
from rllab.misc.resolve import load_class
from rllab.misc.console import mkdir_p, colorize
from rllab.mdp.base import MDP
from rllab.policy.base import Policy
from rllab.vf.base import ValueFunction
from rllab.algo.base import Algorithm
import cPickle as pickle
import os.path as osp
import sys


def instantiate(argvals, cls, *args, **kwargs):
    print colorize('instantiating %s.%s' % (cls.__module__, cls.__name__), 'green')
    return cls.new_from_args(argvals, *args, **kwargs)


def run_interactive():
    pass


if __name__ == "__main__":
    last_args = None
    mkdir_p(osp.expanduser('~/.rllab'))
    cache_file = osp.expanduser('~/.rllab/experiment_args_cache.pkl')
    try:
        with open(cache_file, 'rb') as f:
            last_args = pickle.load(f)
    except Exception:
        pass
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', '-i', action='store_true', help='run in interactive mode')
    parser.add_argument('--algo', type=str, metavar='ALGO_PATH', help='module path to the algorithm')
    parser.add_argument('--mdp', type=str, metavar='MDP_PATH', help='module path to the mdp class')
    # These are optional, depending on the algorithm selected
    parser.add_argument('--policy', type=str, metavar='POLICY_PATH', help='module path to the policy')
    parser.add_argument('--vf', type=str, default='no_value_function', help='module path to the value function')
    parser.add_argument('--more_help', action='store_true', help='whether to show more help depending on the classes chosen')

    args = parser.parse_known_args()[0]

    if args.interactive:
        run_interactive()
    else:
        # Save the arguments which might be useful for later use
        with open(cache_file, 'w+b') as f:
            pickle.dump(sys.argv[1:], f)

        classes = dict()
        classes['mdp'] = load_class(args.mdp, MDP, ["rllab", "mdp"])
        if args.policy:
            classes['policy'] = load_class(args.policy, Policy, ["rllab", "policy"])
        classes['vf'] = load_class(args.vf, ValueFunction, ["rllab", "vf"])
        classes['algo'] = load_class(args.algo, Algorithm, ["rllab", "algo"])

        for cls in classes.values():
            cls.add_args(parser)

        if args.more_help:
            parser.print_help()
            sys.exit(0)

        more_args = parser.parse_known_args()[0]

        instances = dict()
        instances['mdp'] = instantiate(more_args, classes['mdp'])
        if args.policy:
            instances['policy'] = instantiate(more_args, classes['policy'], instances['mdp'])
        instances['vf'] = instantiate(more_args, classes['vf'], instances['mdp'])
        algo = instantiate(more_args, classes['algo'])
        algo.train(**instances)
