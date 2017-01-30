from rllab.sampler.base import BaseSampler
from rllab.misc import tensor_utils
import copy
import numpy as np


class PairwiseSampler(BaseSampler):
    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo

    def start_worker(self):
        self.env0 = copy.deepcopy(self.algo.env)
        self.env1 = copy.deepcopy(self.algo.env)
        # self.agent0 = copy.deepcopy(self.algo.policy)  # needed for recurrence
        # self.agent1 = copy.deepcopy(self.algo.policy)  # but is it enough?

        # self.agent0 = self.algo.policy
        # self.agent1 = self.algo.policy
        # parallel_sampler.populate_task(self.algo.env, self.algo.policy, scope=self.algo.scope)
        pass

    def shutdown_worker(self):
        # parallel_sampler.terminate_task(scope=self.algo.scope)
        pass

    # OLD
    # def obtain_samples(self, itr):
    #     cur_params = self.algo.policy.get_param_values()
    #     paths = parallel_sampler.sample_paths(
    #         policy_params=cur_params,
    #         max_samples=self.algo.batch_size,
    #         max_path_length=self.algo.max_path_length,
    #         scope=self.algo.scope,
    #     )
    #     if self.algo.whole_paths:
    #         return paths
    #     else:
    #         paths_truncated = parallel_sampler.truncate_paths(paths, self.algo.batch_size)
    #         return paths_truncated

    # NEW
    # def obtain_samples(self, itr):
    #     """
    #     This just runs two separate simulations in an alternating fashion,
    #     one step each at a time.
    #     """
    #     paths = []
    #     env0 = self.env0
    #     # env0 = self.algo.env
    #     env1 = self.env1
    #     # agent0 = self.agent0
    #     agent = self.algo.policy
    #     # agent1 = self.agent1

    #     observations0, observations1 = [], []
    #     actions0, actions1 = [], []
    #     rewards0, rewards1 = [], []
    #     agent_infos0, agent_infos1 = [], []
    #     env_infos0, env_infos1 = [], []

    #     o0 = env0.reset()  # OK first question is how to get two of these.
    #     o1 = env1.reset()
    #     # agent0.reset()
    #     # agent1.reset()

    #     # for now, no agent.reset -- FNN only

    #     d0 = False
    #     d1 = False
    #     path_length0 = 0
    #     path_length1 = 0
    #     cum_paths_length = 0
    #     while cum_paths_length < self.algo.batch_size:
    #         if d0 or (path_length0 > self.algo.max_path_length):
    #             # Append result and reset.
    #             paths.append(dict(
    #                 observations=tensor_utils.stack_tensor_list(observations0),
    #                 actions=tensor_utils.stack_tensor_list(actions0),
    #                 rewards=tensor_utils.stack_tensor_list(rewards0),
    #                 agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos0),
    #                 env_infos=tensor_utils.stack_tensor_dict_list(env_infos0),
    #             ))
    #             cum_paths_length += path_length0
    #             observations0 = []
    #             actions0 = []
    #             rewards0 = []
    #             agent_infos0 = []
    #             env_infos0 = []
    #             o0 = env0.reset()
    #             # agent0.reset()
    #             d0 = False
    #             path_length0 = 0
    #         else:
    #             # Or else take a step.
    #             a0, agent_info0 = agent.get_action(o0)
    #             next_o0, r0, d0, env_info0 = env0.step(a0)
    #             observations0.append(env0.observation_space.flatten(o0))
    #             rewards0.append(r0)
    #             actions0.append(env0.action_space.flatten(a0))
    #             agent_infos0.append(agent_info0)
    #             env_infos0.append(env_info0)
    #             o0 = next_o0
    #             path_length0 += 1
    #             # cum_paths_length += 1
    #         if d1 or (path_length1 > self.algo.max_path_length):
    #             paths.append(dict(
    #                 observations=tensor_utils.stack_tensor_list(observations1),
    #                 actions=tensor_utils.stack_tensor_list(actions1),
    #                 rewards=tensor_utils.stack_tensor_list(rewards1),
    #                 agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos1),
    #                 env_infos=tensor_utils.stack_tensor_dict_list(env_infos1),
    #             ))
    #             cum_paths_length += path_length1
    #             observations1 = []
    #             actions1 = []
    #             rewards1 = []
    #             agent_infos1 = []
    #             env_infos1 = []
    #             o1 = env1.reset()
    #             # agent1.reset()
    #             d1 = False
    #             path_length1 = 0
    #         else:
    #             a1, agent_info1 = agent.get_action(o1)
    #             next_o1, r1, d1, env_info1 = env1.step(a1)
    #             observations1.append(env1.observation_space.flatten(o1))
    #             rewards1.append(r1)
    #             actions1.append(env1.action_space.flatten(a1))
    #             agent_infos1.append(agent_info1)
    #             env_infos1.append(env_info1)
    #             o1 = next_o1
    #             path_length1 += 1
    #             # cum_paths_length += 1

    #     return paths

    # NEWER
    # def obtain_samples(self, itr):
    #     """
    #     This one just interleaves the action-requesting with other simulation.
    #     """
    #     paths = []
    #     env0 = self.env0
    #     env1 = self.env1
    #     agent = self.algo.policy
    #     # agent0 = self.agent0
    #     # agent1 = self.agent1

    #     observations0, observations1 = [], []
    #     actions0, actions1 = [], []
    #     rewards0, rewards1 = [], []
    #     agent_infos0, agent_infos1 = [], []
    #     env_infos0, env_infos1 = [], []

    #     o0 = env0.reset()  # OK first question is how to get two of these.
    #     o1 = env1.reset()
    #     # agent0.reset()
    #     # agent1.reset()

    #     d0 = False
    #     d1 = False
    #     path_length0 = 0
    #     path_length1 = 0
    #     cum_paths_length = 0

    #     a0, agent_info0 = agent.get_action(o0)
    #     a1, agent_info1 = agent.get_action(o1)
    #     while cum_paths_length < self.algo.batch_size:
    #         if d0 or (path_length0 > self.algo.max_path_length):
    #             paths.append(dict(
    #                 observations=tensor_utils.stack_tensor_list(observations0),
    #                 actions=tensor_utils.stack_tensor_list(actions0),
    #                 rewards=tensor_utils.stack_tensor_list(rewards0),
    #                 agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos0),
    #                 env_infos=tensor_utils.stack_tensor_dict_list(env_infos0),
    #             ))
    #             cum_paths_length += path_length0
    #             observations0 = []
    #             actions0 = []
    #             rewards0 = []
    #             agent_infos0 = []
    #             env_infos0 = []
    #             o0 = env0.reset()
    #             # agent0.reset()
    #             d0 = False
    #             path_length0 = 0
    #             a0, agent_info0 = agent.get_action(o0)
    #         else:
    #             # a0, agent_info0 = agent0.get_action(o0)
    #             # a0 = np.asarray(a0)  # grab it from GPU
    #             next_o0, r0, d0, env_info0 = env0.step(a0)
    #             observations0.append(env0.observation_space.flatten(o0))
    #             rewards0.append(r0)
    #             actions0.append(env0.action_space.flatten(a0))
    #             # agent_info0 = np.asarray(agent_info0)  # grab from GPU
    #             agent_infos0.append(agent_info0)  # grab it from GPU
    #             env_infos0.append(env_info0)
    #             o0 = next_o0
    #             path_length0 += 1
    #             # cum_paths_length += 1
    #             a0, agent_info0 = agent.get_action(o0)
    #         if d1 or (path_length1 > self.algo.max_path_length):
    #             paths.append(dict(
    #                 observations=tensor_utils.stack_tensor_list(observations1),
    #                 actions=tensor_utils.stack_tensor_list(actions1),
    #                 rewards=tensor_utils.stack_tensor_list(rewards1),
    #                 agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos1),
    #                 env_infos=tensor_utils.stack_tensor_dict_list(env_infos1),
    #             ))
    #             cum_paths_length += path_length1
    #             observations1 = []
    #             actions1 = []
    #             rewards1 = []
    #             agent_infos1 = []
    #             env_infos1 = []
    #             o1 = env1.reset()
    #             # agent1.reset()
    #             d1 = False
    #             path_length1 = 0
    #             a1, agent_info1 = agent.get_action(o1)
    #         else:
    #             # a1, agent_info1 = agent1.get_action(o1)
    #             # a1 = np.asarray(a1)  # grab it from GPU
    #             next_o1, r1, d1, env_info1 = env1.step(a1)
    #             observations1.append(env1.observation_space.flatten(o1))
    #             rewards1.append(r1)
    #             actions1.append(env1.action_space.flatten(a1))
    #             # agent_info1 = np.asarray(agent_info1)  # grab from GPU
    #             agent_infos1.append(agent_info1)
    #             env_infos1.append(env_info1)
    #             o1 = next_o1
    #             path_length1 += 1
    #             # cum_paths_length += 1
    #             a1, agent_info1 = agent.get_action(o1)

    #     return paths


    # # NEWERER
    def obtain_samples(self, itr):
        """
        This one tries to use delayed receipt of GPU results to achieve
        concurrency in getting action with running the other simulator.
        Requires a change in the policy to say don't transfer results.
        """
        paths = []
        env0 = self.env0
        env1 = self.env1
        agent = self.algo.policy
        # agent0 = self.agent0
        # agent1 = self.agent1

        observations0, observations1 = [], []
        actions0, actions1 = [], []
        rewards0, rewards1 = [], []
        agent_infos0, agent_infos1 = [], []
        env_infos0, env_infos1 = [], []

        o0 = env0.reset()  # OK first question is how to get two of these.
        o1 = env1.reset()
        # agent0.reset()
        # agent1.reset()

        d0 = False
        d1 = False
        path_length0 = 0
        path_length1 = 0
        cum_paths_length = 0

        cuda_ref0 = agent.queue_get_action(o0)
        cuda_ref1 = agent.queue_get_action(o1)
        print(type(cuda_ref0[0]))
        print(type(cuda_ref1[1]))
        while cum_paths_length < self.algo.batch_size:
            if d0 or (path_length0 > self.algo.max_path_length):
                paths.append(dict(
                    observations=tensor_utils.stack_tensor_list(observations0),
                    actions=tensor_utils.stack_tensor_list(actions0),
                    rewards=tensor_utils.stack_tensor_list(rewards0),
                    agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos0),
                    env_infos=tensor_utils.stack_tensor_dict_list(env_infos0),
                ))
                cum_paths_length += path_length0
                observations0 = []
                actions0 = []
                rewards0 = []
                agent_infos0 = []
                env_infos0 = []
                o0 = env0.reset()
                # agent0.reset()
                d0 = False
                path_length0 = 0
                cuda_ref0 = agent.queue_get_action(o0)
            else:
                a0, agent_info0 = agent.make_action(cuda_ref0)
                # a0 = np.asarray(a0)  # grab it from GPU
                next_o0, r0, d0, env_info0 = env0.step(a0)
                observations0.append(env0.observation_space.flatten(o0))
                rewards0.append(r0)
                actions0.append(env0.action_space.flatten(a0))
                # agent_info0 = np.asarray(agent_info0)  # grab from GPU
                for k, v in agent_info0.items():
                    agent_info0[k] = np.asarray(v)  # grab it from GPU
                agent_infos0.append(agent_info0)
                env_infos0.append(env_info0)
                o0 = next_o0
                path_length0 += 1
                # cum_paths_length += 1
                cuda_ref0 = agent.queue_get_action(o0)
            if d1 or (path_length1 > self.algo.max_path_length):
                paths.append(dict(
                    observations=tensor_utils.stack_tensor_list(observations1),
                    actions=tensor_utils.stack_tensor_list(actions1),
                    rewards=tensor_utils.stack_tensor_list(rewards1),
                    agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos1),
                    env_infos=tensor_utils.stack_tensor_dict_list(env_infos1),
                ))
                cum_paths_length += path_length1
                observations1 = []
                actions1 = []
                rewards1 = []
                agent_infos1 = []
                env_infos1 = []
                o1 = env1.reset()
                # agent1.reset()
                d1 = False
                path_length1 = 0
                cuda_ref1 = agent.queue_get_action(o1)
            else:
                a1, agent_info1 = agent.make_action(cuda_ref1)
                # a1 = np.asarray(a1)  # grab it from GPU
                next_o1, r1, d1, env_info1 = env1.step(a1)
                observations1.append(env1.observation_space.flatten(o1))
                rewards1.append(r1)
                actions1.append(env1.action_space.flatten(a1))
                # agent_info1 = np.asarray(agent_info1)  # grab from GPU
                for k, v in agent_info1.items():
                    agent_info1[k] = np.asarray(v)
                agent_infos1.append(agent_info1)
                env_infos1.append(env_info1)
                o1 = next_o1
                path_length1 += 1
                # cum_paths_length += 1
                cuda_ref1 = agent.queue_get_action(o1)

        return paths
