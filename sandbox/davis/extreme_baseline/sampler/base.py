from __future__ import print_function
from __future__ import absolute_import
import numpy as np
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger


class Sampler(object):
    def start_worker(self):
        """
        Initialize the sampler, e.g. launching parallel workers if necessary.
        """
        raise NotImplementedError

    def obtain_samples(self, itr):
        """
        Collect samples for the given iteration number.
        :param itr: Iteration number.
        :return: A list of paths.
        """
        raise NotImplementedError

    def process_samples(self, itr, paths):
        """
        Return processed sample data (typically a dictionary of concatenated tensors) based on the collected paths.
        :param itr: Iteration number.
        :param paths: A list of collected paths.
        :return: Processed sample data.
        """
        raise NotImplementedError

    def shutdown_worker(self):
        """
        Terminate workers if necessary.
        """
        raise NotImplementedError


class BaseSampler(Sampler):
    def __init__(self, algo):
        """
        :type algo: BatchPolopt
        """
        self.algo = algo

    def process_samples(self, itr, paths):
        logger.log("computing baselines...")
        baselines = []
        returns = []
        for path in paths:
            path_baselines = np.append(self.algo.baseline.predict(path, self.algo.policy), 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])
        logger.log("done computing baselines")

        if not self.algo.policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if self.algo.center_adv:
                advantages = util.center_advantages(advantages)

            if self.algo.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            ent = np.mean(self.algo.policy.distribution.entropy(agent_infos))

            ev = special.explained_variance_1d(
                np.concatenate(baselines),
                np.concatenate(returns)
            )

            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
            )
        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = np.array([tensor_utils.pad_tensor(ob, max_path_length) for ob in obs])

            if self.algo.center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in actions])

            rewards = [path["rewards"] for path in paths]
            rewards = np.array([tensor_utils.pad_tensor(r, max_path_length) for r in rewards])

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
            )

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
            )

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = np.array([tensor_utils.pad_tensor(v, max_path_length) for v in valids])

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            ent = np.sum(self.algo.policy.distribution.entropy(agent_infos) * valids) / np.sum(valids)

            ev = special.explained_variance_1d(
                np.concatenate(baselines),
                np.concatenate(returns)
            )

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
            )

        logger.record_tabular('Iteration', itr)

        logger.log("fitting baseline...")
        self.algo.baseline.fit(paths, self.algo.policy)
        logger.log("fitted")

        start_state = self.algo.env.wrapped_env.start_state
        policy_params = self.algo.policy.get_param_values()
        mean = policy_params[1] + start_state * policy_params[0]
        stdev = np.exp(policy_params[2])
        lookahead = self.algo.baseline.lookahead
        noise_level = self.algo.env.wrapped_env.noise_level

        correct_constant = -mean**2 * start_state**2 * lookahead - stdev**2
        correct_time_coeff = mean**2 * start_state**2
        correct_noise_coeff = -2 * mean * stdev * start_state
        correct_squared_noise_coeff = -stdev**2
        correct_squared_env_noise_coeff = noise_level
        # All other correct coefficients are zero
        correct_coeffs = np.array([0, 0, correct_time_coeff, 0, 0] +
                                  [correct_noise_coeff for _ in xrange(lookahead - 1)] +
                                  [correct_squared_noise_coeff for _ in xrange(lookahead - 1)] +
                                  [0 for _ in xrange(lookahead - 1)] +
                                  [correct_squared_env_noise_coeff for _ in xrange(lookahead - 1)] +
                                  [correct_constant])

        baseline_params = self.algo.baseline.get_param_values()
        logger.record_tabular("PolicyMean", mean)
        logger.record_tabular("PolicyStdev", stdev)
        logger.record_tabular("BaselineParamNorm", np.linalg.norm(baseline_params))
        logger.record_tabular('DistFromOptBaseline',
                              np.linalg.norm(correct_coeffs - baseline_params))
        if itr == 100: import pdb; pdb.set_trace()

        if 'noise' in env_infos.keys():
            num_rollouts = len(undiscounted_returns)
            path_length = len(env_infos['noise']) / num_rollouts
            # noise_per_rollout = env_infos['noise'].reshape(num_rollouts, path_length, 2)
            # noise_rewards = -np.linalg.norm(noise_per_rollout, axis=2)
            # noise_undiscounted_returns = np.sum(noise_rewards, axis=1)
            # denoised_undiscounted_returns = undiscounted_returns - noise_undiscounted_returns
            # logger.record_tabular('DenoisedUndiscReturn', np.mean(denoised_undiscounted_returns))
            # import pdb; pdb.set_trace()
            actions_per_rollout = actions.reshape(num_rollouts, path_length, -1)
            action_norms = np.linalg.norm(actions_per_rollout, axis=2)
            undisc_action_norms = np.sum(action_norms, axis=1)
            logger.record_tabular('AverageActionNorm', np.mean(undisc_action_norms))

        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data
