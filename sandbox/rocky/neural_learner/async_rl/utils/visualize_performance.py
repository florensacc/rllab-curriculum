import argparse
import os, sys
import joblib
from ..utils.evaluate import evaluate_performance

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file',type=str,default='')
    parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--plot',action="store_true")
    parser.add_argument('--n-runs',type=int,default=1)
    parser.add_argument('--horizon',type=int,default=10**7)
    parser.set_defaults(plot=False)
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print("%s does not exist!"%(args.file))
        sys.exit(0)

    algo = joblib.load(args.file)
    env = algo.cur_env
    if args.plot:
        env.prepare_plot()
    env.phase = "Test"
    agent = algo.cur_agent
    agent.phase = "Test"

    evaluate_performance(env,agent,args.n_runs,args.horizon)
