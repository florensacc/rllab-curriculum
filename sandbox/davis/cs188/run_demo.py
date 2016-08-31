import argparse
from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument("environment",
                    help="which environment to simulate. 1: Half Cheetah. 2: Humanoid.")
parser.add_argument("-n", "--start-itr", default=0,
                    help="which iteration's policy parameters to use for simulation.")
args = parser.parse_args()
directory = {
    '1': "cs188_2016_07_06_00_03_38_0001",
    '2': "PLACEHOLDER"
}[args.environment]
call(['scripts/sim_policy.py', 'data/local/cs188/{}/itr_{}.pkl'.format(directory, args.start_itr)])
