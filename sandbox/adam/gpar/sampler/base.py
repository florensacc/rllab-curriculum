
import numpy as np
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger
from rllab.sampler.utils import rollout
import psutil

import gtimer as gt


class Sampler(object):
    def start_worker(self):
        """
        Initialize the sampler, e.g. launching parallel workers if necessary.
        """
        raise NotImplementedError

    def obtain_samples(self, itr, max_path_length, batch_size):
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


class BaseGpuSampler(Sampler):

    def __init__(self, algo):
        """
        :type algo: GpuBatchPolopt
        """
        self.algo = algo

    def __getstate__(self):
        """ Do not pickle parallel objects """
        return {k: v for k, v in iter(self.__dict__.items()) if k != "par_objs"}

    def initialize_par_objs(self):
        """
        This method should provide, in self.par_objs, at a minimum:
        1. a shutdown signal in 'shutdown',
        2. an iteration barrier at 'barrier',
        3. the multiprocessing processes targeting self.sampling_worker, in
           'workers'.
        Workers must inherit (1) and (2).
        """
        raise NotImplementedError

    def initialize_worker(self, rank, **kwargs):
        """ Any set up for an individual worker once it is spawned """
        pass

    def obtain_samples_worker(self, rank, semapohres, **kwargs):
        """ Worker processes execute this method synchronously with master """
        raise NotImplementedError

    def start_worker(self):
        if not self.initialized:
            self.initialize_par_objs()
        for w in self.par_objs.workers:
            w.start()

    def shutdown_worker(self):
        self.par_objs.shutdown.value = True
        self.par_objs.barrier.wait()
        for w in self.par_objs.workers:
            w.join()

    def sampling_worker(self, rank, semaphores, **kwargs):
        gt.reset_root()
        gt.rename_root('sampler_' + str(rank))
        self.initialize_worker(rank, **kwargs)
        gt.stamp('init_worker')
        loop = gt.timed_loop('main')
        while True:
            next(loop)
            self.par_objs.barrier.wait()
            gt.stamp('outer_barrier')
            if not self.par_objs.shutdown.value:
                self.obtain_samples_worker(rank, semaphores, **kwargs)
                gt.stamp('obtain')
            else:
                break
        loop.exit()
        gt.stop()
        if rank == 0:
            print(gt.report())

    def set_worker_cpu_affinity(self, rank, verbose=True):
        """
        Check your logical cpu vs physical core configuration, use
        cpu_assignments list to put one worker per physical core.  The GPU
        process is pinned to logical CPU 0.
        Default behavior here is to avoid that core for sampling workers
        (including its hyperthread, so for a 4 core, workers go on: 1, 2, 3, 5,
        6, 7, 1, ...)
        """
        if self.cpu_assignments is not None:
            n_assignments = len(self.cpu_assignments)
            assigned_affinity = [self.cpu_assignments[rank % n_assignments]]
        else:
            n_cpu = psutil.cpu_count()
            r_mod = rank % (n_cpu - 2)
            cpu = r_mod + 1
            if cpu >= (n_cpu // 2):
                cpu += 1
            assigned_affinity = [cpu]
        p = psutil.Process()
        try:
            # NOTE: let psutil raise the error if invalid cpu assignment.
            p.cpu_affinity(assigned_affinity)
            if verbose:
                logger.log("Rank: {},  CPU Affinity: {}".format(rank, p.cpu_affinity()))
        except AttributeError:
            logger.log("Cannot set CPU affinity (maybe in a Mac OS).")

    def process_samples(self, itr, paths):
        samples_data, baselines = self.organize_paths(paths)
        self.log_samples_stats(itr, samples_data, baselines)
        self.fit_baseline(paths, samples_data)
        return samples_data

    def organize_paths(self, paths):
        baselines = []
        returns = []

        if hasattr(self.algo.baseline, "predict_n"):
            all_path_baselines = self.algo.baseline.predict_n(paths)
        else:
            all_path_baselines = [self.algo.baseline.predict(path) for path in paths]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.algo.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.algo.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        if not self.algo.policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if self.algo.center_adv:
                advantages = util.center_advantages(advantages)

            if self.algo.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)

        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            observations = tensor_utils.pad_tensor_n(obs, max_path_length)

            if self.algo.center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            advantages = np.asarray([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = tensor_utils.pad_tensor_n(actions, max_path_length)

            rewards = [path["rewards"] for path in paths]
            rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

            returns = [path["returns"] for path in paths]
            returns = tensor_utils.pad_tensor_n(returns, max_path_length)

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
            )

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
            )

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = tensor_utils.pad_tensor_n(valids, max_path_length)

        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            returns=returns,
            advantages=advantages,
            env_infos=env_infos,
            agent_infos=agent_infos,
            paths=paths,
        )
        if self.algo.policy.recurrent:
            samples_data["valids"] = valids

        return samples_data, baselines

    def log_samples_stats(self, itr, samples_data, baselines):
        returns = samples_data["returns"]
        paths = samples_data["paths"]
        agent_infos = samples_data["agent_infos"]

        ev = special.explained_variance_1d(np.concatenate(baselines), returns)
        average_discounted_return = \
            np.mean([path["returns"][0] for path in paths])
        undiscounted_returns = [sum(path["rewards"]) for path in paths]
        if not self.algo.policy.recurrent:
            ent = np.mean(self.algo.policy.distribution.entropy(agent_infos))
        else:
            valids = samples_data["valids"]
            ent = np.sum(self.algo.policy.distribution.entropy(agent_infos) * valids) / np.sum(valids)

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular_misc_stat('TrajLen', [len(p["rewards"]) for p in paths], placement='front')
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular_misc_stat('Return', undiscounted_returns, placement='front')

    def fit_baseline(self, paths, samples_data):
        logger.log("fitting baseline...")
        if hasattr(self.algo.baseline, 'fit_with_samples'):
            self.algo.baseline.fit_with_samples(paths, samples_data)
        else:
            self.algo.baseline.fit(paths)
        logger.log("fitted")

    def obtain_samples_example(self, path_length=10, num_paths=2):
        paths = []
        for _ in range(num_paths):
            paths.append(
                rollout(self.algo.env, self.algo.policy, max_path_length=path_length))
        self.algo.env.reset()
        self.algo.policy.reset()
        return paths
