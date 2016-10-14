from rllab.core.serializable import Serializable
from rllab.sampler.utils import rollout

import numpy as np
from rllab.misc import tensor_utils
import time


class MixtureDemoCollector(Serializable):
    def __init__(self, policy_cls):
        Serializable.quick_init(self, locals())
        self.policy_cls = policy_cls

    def collect_demo(self, env, horizon, novice):
        policy = self.policy_cls(env)
        #return rollout(env, policy, max_path_length=horizon)
        return mixture_rollout(env, policy, novice, horizon, animated=True)


def mixture_rollout(env, expert_agent, novice_agent, max_path_length=np.inf, animated=False, speedup=1):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    expert_agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        novice_a, agent_info = novice_agent.get_action(o)
        expert_a, agent_info = expert_agent.get_action(o)
        choice = np.random.randint(0, 1)
        if choice == 0:
            true_a = expert_a
        else:
            true_a = novice_a
        next_o, r, d, env_info = env.step(true_a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(expert_a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
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
    )


if __name__ == "__main__":

    from sandbox.bradly.analogy.policy.conopt_particle_tracking_policy import ConoptParticleTrackingPolicy
    from sandbox.bradly.analogy.envs.conopt_particle_env import ConoptParticleEnv

    env = ConoptParticleEnv()
    pol = ConoptParticleTrackingPolicy(env)
    p = PolicyDemoCollector(ConoptParticleTrackingPolicy, pol)
    p.collect_demo(env, 100)