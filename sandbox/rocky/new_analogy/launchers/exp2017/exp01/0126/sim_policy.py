import joblib
import numpy as np
import tensorflow as tf

if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument('file', type=str,
    #                     help='path to the snapshot file')
    # parser.add_argument('--max_path_length', type=int, default=1000,
    #                     help='Max length of rollout')
    # parser.add_argument('--speedup', type=float, default=1,
    #                     help='Speedup')
    # parser.add_argument('--deterministic', action="store_true", help='Deterministic')
    # args = parser.parse_args()

    policy = None
    env = None

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    # while True:
    with tf.Session() as sess:
        data = joblib.load("data/resource/fetch_5_blocks_dagger.pkl")#args.file)
        policy = data['policy']
        env = data['env']

        ob = env.reset()

        while True:
            env.render()
            a, info = policy.get_action(ob)
            ob, _, _, _ = env.step(a)

            for pid in range(4):
                prob = info['id_{}_prob'.format(pid)]
                ent = np.sum(-prob * np.log(prob+1e-8))
                print("Id:",pid,"Ent:",ent,"Perplexity:",np.exp(ent))



        # if args.deterministic:
        # policy = DeterministicPolicy(env_spec=env.spec, wrapped_policy=policy)
        # while True:
        #     path = rollout(env, policy, max_path_length=args.max_path_length,
        #                    animated=True, speedup=args.speedup)
        #     print(path["rewards"][-1])