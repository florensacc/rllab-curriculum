import argparse
import joblib
import uuid
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sandbox.haoran.myscripts.myutilities import get_true_env
from gym.envs.mujoco import mujoco_env
from rllab.envs.proxy_env import ProxyEnv
from sandbox.tuomas.mddpg.policies.stochastic_policy import StochasticNNPolicy
from rllab.misc.ext import set_seed
from rllab.misc import tensor_utils
import time

def rollout(sess, env, agent, max_path_length=np.inf, animated=False, speedup=1, qf=None):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0

    true_env = env
    while isinstance(env, ProxyEnv):
        env = env._wrapped_env

    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        #a = 0 * a

        #pos = env.model.data.qpos.copy().squeeze()
        #vel = env.model.data.qvel.copy().squeeze()
        #pos[3] -= 0.1

        #import pdb; pdb.set_trace()

        #env.set_state(pos, vel)

        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            if qf is not None:
                feed = {
                    qf.observations_placeholder: np.array([o]),
                    qf.actions_placeholder: np.array([a]),
                }
                qvalue = sess.run(qf.output, feed).ravel()
                print("terminate value:", qvalue - r)
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated:
        env.render(close=True)

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
    parser.add_argument('--seed', type=int, default=-1,
        help='Fixed random seed for each rollout. Set for reproducibility.')
    parser.add_argument('--no-plot', default=False, action='store_true')
    parser.add_argument('--show-qf', default=False, action='store_true')
    parser.add_argument('--plt-backend', type=str, default="")
    args = parser.parse_args()

    qf = None
    with tf.Session() as sess:
        data = joblib.load(args.file)
        if "algo" in data:
            algo = data["algo"]
            if hasattr(algo, "eval_policy") and \
                (algo.eval_policy is not None):
                policy = algo.eval_policy
            else:
                policy = algo.policy
            env = data["algo"].env
            if args.show_qf:
                qf = data["algo"].qf
        else:
            policy = data['policy']
            env = data['env']
            if args.show_qf:
                qf = data['qf']
        while True:
            if args.plt_backend != "":
                plt.switch_backend('MacOSX')
            if args.seed >= 0:
                set_seed(args.seed)
                true_env = get_true_env(env)
                if isinstance(true_env, mujoco_env.MujocoEnv):
                    # Gym mujoco env doesn't use the default np seed
                    true_env._seed(args.seed)
            path = rollout(sess, env, policy, max_path_length=args.max_path_length,
                           animated=(not args.no_plot), speedup=args.speedup,
                           qf=qf)
