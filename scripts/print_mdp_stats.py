import argparse
from rllab.mdp.base import MDP
from rllab.misc.resolve import load_class

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mdp', type=str, help='module path to the mdp class')
    args = parser.parse_args()
    mdp = load_class(args.mdp, MDP, ["rllab", "mdp"])()
    mdp.print_stats()
