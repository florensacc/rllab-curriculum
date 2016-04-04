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
    args = parser.parse_args()

    data = joblib.load(args.file)
    policy = data['policy']
    env = data['env']
    paths = data.get('paths', None)
    import ipdb; ipdb.set_trace()
