from rllab.sampler.utils import rollout
import argparse
import joblib
import uuid
import os

filename = str(uuid.uuid4())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=int, default=1,
                        help='Speedup')
    parser.add_argument('--loop', type=int, default=1,
                        help='# of loops')
    args = parser.parse_args()

    policy = None
    mdp = None
    while True:
        if ':' in args.file:
            # fetch file using ssh
            os.system("rsync -avrz %s /tmp/%s.pkl" % (args.file, filename))
            data = joblib.load("/tmp/%s.pkl" % filename)
            if policy:
                new_policy = data['policy']
                policy.set_param_values(new_policy.get_param_values())
                path = rollout(mdp, policy, max_length=args.max_length,
                               animated=True, speedup=args.speedup)
            else:
                policy = data['policy']
                mdp = data['mdp']
                mdp.start_viewer()
                path = rollout(mdp, policy, max_length=args.max_length,
                               animated=True, speedup=args.speedup)
        else:
            data = joblib.load(args.file)
            policy = data['policy']
            mdp = data['mdp']
            mdp.start_viewer()
            path = rollout(mdp, policy, max_length=args.max_length,
                           animated=True, speedup=args.speedup)
            mdp.stop_viewer()
        # print 'Total reward: ', sum(path["rewards"])
        args.loop -= 1
        if ':' not in args.file:
            if args.loop <= 0:
                break
