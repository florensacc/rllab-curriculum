import numpy as np
import sys
sys.path.append('.')
from myscripts.myutilities import load_problem
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dir',type=str,default='',nargs='?')
parser.add_argument('--file',type=str,default='params.pkl',nargs='?')
parser.add_argument('--iteration',type=int,default= -1,nargs='?')
parser.add_argument('--speedup',type=float,default=5,nargs='?')
args = parser.parse_args()

if args.iteration >= 0:
    problem = load_problem(args.dir,iteration=args.iteration)
else:
    problem = load_problem(args.dir,pkl_file=args.file)
algo = problem["algo"]
env = problem["env"]
policy = problem["policy"]

import rllab.plotter as plotter
plotter.init_worker()
plotter.init_plot(env,policy)
plotter.update_plot(policy,algo.max_path_length)

from rllab.sampler.utils import rollout
while True:
    import pdb; pdb.set_trace()
    rollout(env, policy, max_path_length=algo.max_path_length, animated=True, speedup=args.speedup)
