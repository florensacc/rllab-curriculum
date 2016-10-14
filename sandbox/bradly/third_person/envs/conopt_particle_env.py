import logging
log_level = logging.getLogger().level
import gym
logging.getLogger().setLevel(log_level)
from rllab.envs.base import Env, Step
from rllab.spaces.box import Box
from rllab.core.serializable import Serializable
from conopt.experiments.A8_particle_imitation import Experiment
import numpy as np
from sandbox.rocky.analogy.utils import unwrap, using_seed
from rllab.envs.gym_env import convert_gym_space
from conopt import cost
from cached_property import cached_property
from conopt.worldgen.objs import Obj


def fast_residual2cost(r, metric):
    if len(r.shape) == 1:
        r = np.expand_dims(r, 0)
    if metric == "L2":
        return 0.5 * np.sum(np.square(r))
    else:
        import ipdb;
        ipdb.set_trace()


def fast_compute_cost(reward_fn, s):
    if isinstance(reward_fn, cost.MulCost):
        return reward_fn.b * fast_compute_cost(reward_fn.a_cost, s)
    elif isinstance(reward_fn, cost.AddCost):
        return fast_compute_cost(reward_fn.a_cost, s) * fast_compute_cost(reward_fn.b_cost, s)
    elif isinstance(reward_fn, cost.DistCost):
        return fast_residual2cost(s[reward_fn.a] - s[reward_fn.b], reward_fn.metric)
    elif isinstance(reward_fn, cost.PenaltyCost):
        return fast_residual2cost(s[reward_fn.element], reward_fn.metric)
    else:
        import ipdb;
        ipdb.set_trace()


class ConoptParticleEnv(Env, Serializable):
    def __init__(self, seed=None, target_seed=None, obs_type='state'):
        Serializable.quick_init(self, locals())
        self.seed = seed
        self.target_seed = target_seed
        self.conopt_exp = None
        self.conopt_scenario = None
        self.conopt_env = None
        self.curr_target_idx = 0
        self.reset_trial()

    def reset_trial(self):
        seed = np.random.randint(np.iinfo(np.int32).max)
        self.seed = seed
        target_seed = np.random.randint(np.iinfo(np.int32).max)
        self.target_seed = target_seed
        exp = Experiment()
        with using_seed(self.seed):
            scenario = exp.make_scenario(trial_index=seed)
        env = scenario.to_env()
        self.conopt_exp = exp
        self.conopt_scenario = scenario
        self.conopt_env = env
        return self.reset()

    def reset(self):
        return self.conopt_env.reset()

    @cached_property
    def observation_space(self):
        return convert_gym_space(self.conopt_env.observation_space)

    @cached_property
    def action_space(self):
        bounds = self.model.actuator_ctrlrange
        lb = bounds[:, 0]
        ub = bounds[:, 1]
        return Box(lb, ub)

    def render(self, *args, **kwargs):
        self.conopt_env.render(*args, **kwargs)

    @cached_property
    def model(self):
        return self.conopt_env.world.model.model

    def step(self, action):
        env = self.conopt_env
        action = action.reshape(env.action_space.shape)
        next_obs, rew, done, infos = env.step(action)
        return Step(next_obs, rew, done)

