from rllab.core.serializable import Serializable
#from sandbox.rocky.analogy.envs.conopt_particle_env import ConoptParticleEnv
#from sandbox.bradly.analogy.envs.conopt_floating_arm import ConoptArmEnv
from sandbox.bradly.third_person.envs.conopt_particle_env_easy import ConoptParticleEnvEasy
from sandbox.rocky.analogy.utils import unwrap
from conopt.trajectory import Trajectory
from conopt.core import OptimizerParams

import numpy as np


class LQRExpert(Serializable):
    def __init__(self, optimizer_params=None):
        Serializable.quick_init(self, locals())
        self.optimizer_params = optimizer_params

    def get_expert_policy(self, env, horizon):
        pass

    def do_expert_iteration(self, env, horizon):
        pass

    def collect_expert_demo(self, env, horizon):
        env = unwrap(env)
        assert isinstance(env, ConoptParticleEnvEasy)
        scenario = env.conopt_scenario
        trajopt = Trajectory(scenario=scenario)
        opt_params = OptimizerParams(T=horizon, num_iterations=2, **(self.optimizer_params or dict()))
        trajopt.optimize(opt_params)
        observations = env.observation_space.flatten_n(list(zip(*trajopt.observations_over_time())))
        actions = env.action_space.flatten_n(trajopt.actions_over_time())
        rewards = - trajopt.solution['cost'][:horizon].flatten()
        #samp = trajopt.sampleRollout()
        #print(samp)
        #asfasf
        return dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            agent_infos=dict(),
            env_infos=dict()
        )


if __name__ == '__main__':
    num_expert_trajs = 100
    #for iter_step in range(0, num_expert_trajs):
    t = LQRExpert()
    bill = ConoptParticleEnvEasy()
    tim = t.collect_expert_demo(bill, horizon=10)
    r = LQRExpert()
    tim_2 = t.collect_expert_demo(bill, horizon=10)
    #print(tim['rewards'])
    #print(iter_step)
    assert np.allclose(tim['rewards'], tim_2['rewards'])
