# from rllab.sampler.utils import rollout
import argparse
import joblib
import uuid

filename = str(uuid.uuid4())

import numpy as np
from rllab.misc import tensor_utils
import time


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1, n_seq_frames=1, model=None):
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
    obs_bfr = np.zeros((n_seq_frames, env.spec.observation_space.shape[1]))
    # We fill up with the starting frame, better than black.
    for i in range(n_seq_frames):
        obs_bfr[i] = o
    while path_length < max_path_length:
        obs_bfr[top_ptr] = o
        # last frame [-1]
        o = np.concatenate((obs_bfr[(top_ptr + 1) % n_seq_frames:],
                            obs_bfr[:(top_ptr + 1) % n_seq_frames]), axis=0)
        # format for policy/baseline is w x h x n_samp
        # o = o.transpose((2, 1, 0))
        # Split RAM from image
        a, agent_info = agent.get_action(o[:, :128])
        next_o, r, d, env_info = env.step(a)
        env_info['images'] = o[:, 128:].flatten()
        keys = obs_to_key(env_info['images'][None,:])
        string = str(np.array_str(keys[0], max_line_width=1000000)).replace('0', ' ').replace('1', u"\u2588")[1:-1]
        print(string)

        observations.append(env.observation_space.flatten(o[:, :128]))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
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


def encode_obs(obs, model):
    """
    Observation into uint8 encoding, also functions as target format
    """
    assert np.max(obs) <= 1.0
    assert np.min(obs) >= -1.0
    obs_enc = np.floor((obs + 1.0) * 0.5 * model.num_classes).astype("uint8")
    return obs_enc


def decode_obs(obs, model):
    """
    From uint8 encoding to original observation format.
    """
    obs_dec = obs / float(model.num_classes) * 2.0 - 1.0
    assert np.max(obs_dec) <= 1.0
    assert np.min(obs_dec) >= -1.0
    return obs_dec

def obs_to_key(image):
    # Encode/decode to get uniform representation.
    # FIXME: to img change
    obs_ed = decode_obs(encode_obs(image, model), model)
    # obs_ed = self.decode_obs(
    #     self.encode_obs(path['observations'][:, -np.prod(self._model.state_dim):]))
    # Get continuous embedding.
    cont_emb = model.discrete_emb(obs_ed)
    # Cast continuous embedding into binary one.
    # return np.cast['int'](np.round(cont_emb))
    bin_emb = np.cast['int'](np.round(cont_emb))
    bin_emb_downsampled = bin_emb.reshape(-1, 8).mean(axis=1).reshape((bin_emb.shape[0], -1))
    return np.cast['int'](np.round(bin_emb_downsampled))





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
        model = data['model']
        while True:
            path = rollout(env, policy, max_path_length=args.max_path_length,
                           animated=True, speedup=args.speedup, model=model)
