
import numpy as np

from rllab.misc import special, tensor_utils
from rllab.algos import util
from rllab.sampler.utils import rollout


class WorkerBatchSampler(object):

    def __init__(self, algo):
        """
        :type algo: ParallelBatchPolopt
        """
        self.algo = algo
        self.worker_batch_size = algo.worker_batch_size
        self.n_steps_collected = 0

    def obtain_samples(self, n_samples=None):
        if n_samples is None:
            n_samples = self.worker_batch_size
        n_steps_collected = 0
        paths = []
        # TODO: progbar for rank 0?
        while n_steps_collected < n_samples:
            paths.append(rollout(self.algo.env, self.algo.policy, self.algo.max_path_length))
            n_steps_collected += len(paths[-1]["rewards"])
        if self.algo.whole_paths:
            self.algo.n_steps_collected = n_steps_collected
            return paths
        else:
            paths_truncated = self._truncate_paths(paths)
            return paths_truncated

    def _truncate_paths(self, paths):
        """
        Truncate the list of paths so that the total number of samples is exactly
        equal to worker_batch_size. This is done by removing extra paths at the end
        of the list, and make the last path shorter if necessary
        :param paths: a list of paths
        :return: a list of paths, truncated so that the number of samples adds up to max-samples
        """
        # chop samples collected by extra paths
        # make a copy
        paths = list(paths)
        total_n_samples = sum(len(path["rewards"]) for path in paths)
        while len(paths) > 0 and total_n_samples - len(paths[-1]["rewards"]) >= self.worker_batch_size:
            total_n_samples -= len(paths.pop(-1)["rewards"])
        if len(paths) > 0:
            last_path = paths.pop(-1)
            truncated_last_path = dict()
            truncated_len = len(last_path["rewards"]) - (total_n_samples - self.worker_batch_size)
            for k, v in last_path.items():
                if k in ["observations", "actions", "rewards"]:
                    truncated_last_path[k] = tensor_utils.truncate_tensor_list(v, truncated_len)
                elif k in ["env_infos", "agent_infos"]:
                    truncated_last_path[k] = tensor_utils.truncate_tensor_dict(v, truncated_len)
                else:
                    raise NotImplementedError
            paths.append(truncated_last_path)
        return paths

    def process_samples(self, paths):
        baselines = []
        returns = []
        for path in paths:
            path_baselines = np.append(self.algo.baseline.predict(path), 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        dgnstc_data = dict(baselines=baselines, returns=returns)

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

            # NOTE: Removed diagnostics calculation to batch_polopt.

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

            # NOTE: Removed diagnostics calculation to batch_polopt.

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

        # NOTE: Removed baseline fitting to batch_polopt.
        return samples_data, dgnstc_data
