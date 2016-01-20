from rllab.sampler.utils import rollout
import argparse
import joblib
from rllab.mdp.john_mjc import SwimmerMDP

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_length', type=int, default=1000,
                        help='Max length of rollout')

    args = parser.parse_args()
    data = joblib.load(args.file)
    policy = data['policy']
    from rllab.mdp.mujoco_pre_2.cheetah_mdp import CheetahMDP
    mdp = SwimmerMDP()#data['mdp']
    mdp.start_viewer()
    path = rollout(mdp, policy, max_length=args.max_length, animated=True)
    mdp.stop_viewer()
    print 'Total reward: ', sum(path["rewards"])
