import numpy as np


def rollout(env, policy, max_path_length):

    observations, rewards, actions = [], [], []
    o = env.reset()
    for t in range(max_path_length):
        a, _ = policy.get_action(o)
        o, r, d, _ = env.step(a)

        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))

        if d:
            break

    return dict(observations=np.array(observations),
                actions=np.array(actions),
                rewards=np.array(rewards))
