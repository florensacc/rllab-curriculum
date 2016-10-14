from rllab.core.serializable import Serializable
from sandbox.rocky.analogy.envs.conopt_particle_env import ConoptParticleEnv
from sandbox.rocky.analogy.utils import unwrap


class TrajoptDemoCollector(Serializable):
    def __init__(self, optimizer_params=None):
        Serializable.quick_init(self, locals())
        self.optimizer_params = optimizer_params

    def collect_demo(self, env, horizon):
        from conopt.trajectory import Trajectory
        from conopt.core import OptimizerParams
        env = unwrap(env)
        assert isinstance(env, ConoptParticleEnv)
        scenario = env.conopt_scenario
        trajopt = Trajectory(scenario=scenario)
        opt_params = OptimizerParams(T=horizon, **(self.optimizer_params or dict()))
        trajopt.optimize(opt_params)
        observations = env.observation_space.flatten_n(list(zip(*trajopt.observations_over_time())))
        actions = env.action_space.flatten_n(trajopt.actions_over_time())
        rewards = - trajopt.solution['cost'][:horizon].flatten()
        return dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            agent_infos=dict(),
            env_infos=dict()
        )
