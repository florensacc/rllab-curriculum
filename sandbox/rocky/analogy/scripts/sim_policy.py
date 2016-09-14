from rllab.sampler.utils import rollout
import argparse
import joblib
import uuid
import tensorflow as tf
import numpy as np
from sandbox.rocky.analogy.policies.apply_demo_policy import ApplyDemoPolicy

filename = str(uuid.uuid4())

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    args = parser.parse_args()

    policy = None
    env = None

    while True:
        with tf.Session() as sess:
            data = joblib.load(args.file)
            policy = data['policy']
            trainer = data['trainer']
            env_cls = trainer.env_cls

            while True:
                target_seed = np.random.randint(np.iinfo(np.int32).max)
                seed = np.random.randint(np.iinfo(np.int32).max)
                demo_env = env_cls(seed=seed, target_seed=target_seed)
                analogy_env = env_cls(seed=seed + 10007, target_seed=target_seed)
                demo_policy = trainer.demo_policy_cls(demo_env)

                demo_path = rollout(demo_env, demo_policy, max_path_length=trainer.horizon, animated=True,
                                    speedup=args.speedup)
                path = rollout(analogy_env, ApplyDemoPolicy(policy, demo_path),
                               max_path_length=trainer.horizon,
                               animated=True, speedup=args.speedup)
