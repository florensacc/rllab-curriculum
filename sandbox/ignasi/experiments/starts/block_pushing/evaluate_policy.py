



import numpy as np
from rllab.misc import tensor_utils
import time



import argparse

import joblib
import tensorflow as tf

from rllab.misc.ext import set_seed

from sandbox.dave.rllab.goal_generators.pr2_goal_generators import PR2CrownGoalGeneratorSmall, PR2FixedGoalGenerator #PR2CrownGoalGeneratorSmall
from sandbox.dave.rllab.lego_generators.pr2_lego_generators import PR2LegoBoxBlockGeneratorSmall, PR2LegoBoxBlockGeneratorSmall,PR2LegoBoxBlockGeneratorSmall, PR2LegoFixedBlockGenerator

import time
def block_pushing_visualize_rollouts(env, agent, lego_x, lego_y, max_path_length=np.inf, animated=False, speedup=1, no_action=True, repeat = 1):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    dones = []

    goal_generator = PR2FixedGoalGenerator(
        goal=(0.6, 0.1, 0.5025))  # second dimension moves block further away vertically
    init_hand = np.array([0.6, 0.2, 0.5025])

    # generate blocks at many different positions
    lego_generator = PR2LegoFixedBlockGenerator(block=(0.6 +lego_x * 0.01, 0.1 + lego_y * 0.01, 0.5025, 1, 0, 0, 0))
    print("Generated lego position: " + str(lego_generator.generate_goal(0))) #0 doesn't matter
    base_env = env
    while hasattr(base_env, 'wrapped_env'):
        base_env = base_env.wrapped_env
    base_env.goal_generator = goal_generator
    base_env._lego_generator = lego_generator
    base_env.fixed_target = init_hand
    base_env.no_action = no_action

    o = env.reset()
    agent.reset()
    # env.no_action = no_action
    print("Initial tip position: " + str(base_env.get_tip_position()))
    print("Initial lego position: " + str(base_env.get_lego_position()))
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        # if no_action:
        #     a = np.zeros_like(a)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        dones.append(d)
        path_length += 1
        if d:
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
        dones=np.asarray(dones),
        last_obs=o,
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
                        help='Fixed random seed')
    parser.add_argument('--mode', type=str, default="check_boundaries",
                        help='whether to check initial positoin or to view policies')
    args = parser.parse_args()

    policy = None
    env = None

    if args.mode == "check_boundaries":
        if args.seed >= 0:
            set_seed(args.seed)
        with tf.Session() as sess:
            data = joblib.load(args.file)
            if "algo" in data:
                policy = data["algo"].policy
                env = data["algo"].env
            else:
                policy = data['policy']
                env = data['env']
            for lego_x in range(-20, 21, 2):
                for lego_y in range(-40, 41, 2):
                    path = block_pushing_visualize_rollouts(env, policy, max_path_length=2,
                                   animated=True, speedup=args.speedup, lego_x=lego_x, lego_y=lego_y)
                    # print(path["rewards"][-1])
                    time.sleep(0.1)
    else:
        pass