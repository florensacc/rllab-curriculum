from rllab.core.serializable import Serializable
from rllab.envs.proxy_env import ProxyEnv


class StopActionEnv(ProxyEnv, Serializable):
    def __init__(
            self,
            env
    ):
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)

    def step(self, action):
        observation, reward, done, info = ProxyEnv.step(self, action)
        info['reward_inner'] = reward_inner = self.inner_weight * reward
        info['distance'] = dist = self.dist_to_goal(observation)
        info['reward_dist'] = reward_dist = self.compute_dist_reward(observation)
        info['goal_reached'] = 1.0 * self.is_goal_reached(observation)
        info['goal'] = self.current_goal
        # print(reward_dist)
        # print(reward_inner)
        # print("step: obs={}, goal={}, dist={}".format(self.append_goal_observation(observation), self.current_goal, dist))
        if self.terminate_env and self.is_goal_reached(observation):  # if the inner env is done it will stay done.
            # print("\n*******done**********\n")
            # print("the dist in the goal env is: ", dist)
            done = True
        return (
            self.append_goal_observation(observation),
            reward_dist + reward_inner,
            done,
            info
        )