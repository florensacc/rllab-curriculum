from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize
from rllab.misc import tensor_utils
from sandbox.haoran.myscripts.envs import EnvChooser
import numpy as np
import time


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1,
    config=None):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render(config)
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render(config)
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated:
        env.render(close=True, config=config)

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )

# ----------------------------------------------------------------------------
def run():
    true_env = EnvChooser().choose_env("multilink_reacher")
    # true_env = BilliardsEnv(
    #     random_init_state=False,
    # )
    env = normalize(
        true_env,
        normalize_obs=True,
    )

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(32, 32),
        init_std=1,
    )
    true_env.window_config=dict(
        xpos=0,
        ypos=0,
        width=500,
        height=500,
        title="simulation",
    )

    while True:
        rollout(env, policy, max_path_length=np.inf, animated=True, speedup=1)

if __name__ == "__main__":
    run()
