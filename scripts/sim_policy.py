# from rllab.sampler.utils import rollout
import argparse
import joblib
import uuid
import os
import random
import numpy as np
import tensorflow as tf

filename = str(uuid.uuid4())

import numpy as np
from rllab.misc import tensor_utils
import time
from sandbox.rein.algos.embedding_theano_par.plotter import Plotter


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1, n_seq_frames=4, model=None):
    def encode_obs(obs):
        """
        Observation into uint8 encoding, also functions as target format
        """
        assert np.max(obs) <= 1.0
        assert np.min(obs) >= -1.0
        obs_enc = np.round((obs + 1.0) * 0.5 * (model.num_classes - 1)).astype("uint8")
        return obs_enc

    def decode_obs(obs):
        """
        From uint8 encoding to original observation format.
        """
        obs_dec = obs / float(model.num_classes - 1) * 2.0 - 1.0
        assert np.max(obs_dec) <= 1.0
        assert np.min(obs_dec) >= -1.0
        return obs_dec

    print("FIXME")
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    # buffer of empty frames.
    top_ptr = 0
    obs_bfr = np.zeros((n_seq_frames, env.spec.observation_space.shape[2], env.spec.observation_space.shape[1]))
    # We fill up with the starting frame, better than black.
    for i in range(n_seq_frames):
        obs_bfr[i] = o
    plotter = Plotter()
    count = 0
    while path_length < max_path_length:
        obs_bfr[top_ptr] = o
        # last frame [-1]
        o = np.concatenate((obs_bfr[(top_ptr + 1) % n_seq_frames:],
                            obs_bfr[:(top_ptr + 1) % n_seq_frames]), axis=0)
        # format for policy/baseline is w x h x n_samp
        # o = o.transpose((2, 1, 0))
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1

        oo = encode_obs(env.observation_space.flatten(o)[-np.prod(model.state_dim):][None, :])
        # plotter.plot_pred_imgs(
        #     model=model, inputs=decode_obs(oo), targets=oo,
        #     itr=count, dir='tmp/pred')
        # plotter.plot_actual_imgs(
        #     inputs=o[-np.prod(model.state_dim):],
        #     itr=count, dir='tmp/actual')
        plotter.print_consistency_embs(
            model=model, projection_matrix=None,
            inputs=decode_obs(oo),
            dir='tmp/pred', hamming_distance=0)
        count += 1
        if d:
            break
        o = next_o
        top_ptr = (top_ptr + 1) % n_seq_frames
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )


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

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]
    while True:
        # with tf.Session() as sess:
        data = joblib.load(args.file)
        policy = data['policy']
        env = data['env']
        print('loaded')
        while True:
            path = rollout(env, policy, max_path_length=args.max_path_length,
                           animated=True, speedup=args.speedup, model=data['model'])
            print(sum(path['rewards']))
