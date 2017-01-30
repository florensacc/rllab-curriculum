
import numpy as np
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
import rllab.misc.logger as logger
from rllab.misc import ext
from rllab.sampler.utils import rollout
import psutil
from sandbox.adam.util import struct
import multiprocessing as mp
from ctypes import c_bool
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

    def __init__(self, algo, n_parallel):
        """
        :type algo: GpuBatchPolopt
        """
        self.algo = algo
        self.n_parallel = n_parallel

    def __getstate__(self):
        """ Do not pickle parallel objects """
        return {k: v for k, v in iter(self.__dict__.items()) if k != "par_objs"}

    def initialize_par_objs(self):
        """
        Return any shared objects to pass to workers (avoiding inheritance in
        order to limit which workers have each shared variable on an as-needed
        basis).
        Must be a tuple: first is par_objs for master, second is list of
        par_objs for each rank.
        """
        return (None, [None] * (self.n_parallel))

    def initialize_worker(self, rank):
        """ Any set up for an individual worker once it is spawned """
        if self.set_cpu_affinity:
            self.set_worker_cpu_affinity(rank)
        seed = ext.get_seed()
        if seed is None:
            # NOTE: not sure if this is a good source for seed?
            seed = int(1e6 * np.random.rand())
        ext.set_seed(seed + rank)
        self.initialize_worker_extra(rank)  # anything else to be done

    def initialize_worker_extra(self, rank):
        """ Can be used by derivative classes """
        pass

    def start_worker(self):
        par_objs_master, par_objs_ranks = self.initialize_par_objs()
        loop_ctrl = struct(
            shutdown=mp.RawValue(c_bool, False),
            barrier=mp.Barrier(self.n_parallel + 1),
        )
        workers = []
        for rank in range(self.n_parallel):
            workers.append(mp.Process(target=self.sampling_worker,
                args=(rank, loop_ctrl, par_objs_ranks[rank])))
        for w in workers:
            w.start()
        par_objs_master.loop_ctrl = loop_ctrl  # (don't use these two fields)
        par_objs_master.workers = workers
        return par_objs_master  # (don't attach to self yet, other GPU processes don't need it)

    def shutdown_worker(self):
        self.par_objs.loop_ctrl.shutdown.value = True
        self.par_objs.loop_ctrl.barrier.wait()
        for w in self.par_objs.workers:
            w.join()

    def obtain_samples_worker(self, rank, semapohres, **kwargs):
        """ Worker processes execute this method synchronously with master """
        raise NotImplementedError

    def sampling_worker(self, rank, loop_ctrl, par_objs_rank):
        gt.reset_root()
        gt.rename_root('sampler_' + str(rank))
        self.par_objs = par_objs_rank  # make par_objs accessible elsewhere in worker
        self.initialize_worker(rank)
        gt.stamp('init_worker')
        loop = gt.timed_loop('main')
        while True:
            next(loop)
            loop_ctrl.barrier.wait()
            gt.stamp('outer_barrier')
            if not loop_ctrl.shutdown.value:
                self.obtain_samples_worker(rank)
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

    # def fit_baseline(self, paths, samples_data):
    #     logger.log("fitting baseline...")
    #     if hasattr(self.algo.baseline, 'fit_with_samples'):
    #         self.algo.baseline.fit_with_samples(paths, samples_data)
    #     else:
    #         self.algo.baseline.fit(paths)
    #     logger.log("fitted")

    def obtain_samples_example(self, path_length=10, num_paths=2):
        paths = []
        for _ in range(num_paths):
            paths.append(
                rollout(self.algo.env, self.algo.policy, max_path_length=path_length))
        self.algo.env.reset()
        self.algo.policy.reset()
        return paths

    def process_samples(self, itr, paths):
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

        ev = special.explained_variance_1d(
            np.concatenate(baselines),
            np.concatenate(returns)
        )

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

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            ent = np.mean(self.algo.policy.distribution.entropy(agent_infos))

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
        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = tensor_utils.pad_tensor_n(obs, max_path_length)

            if self.algo.center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.asarray([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

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

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            undiscounted_returns = [sum(path["rewards"]) for path in paths]

            ent = np.sum(self.algo.policy.distribution.entropy(agent_infos) * valids) / np.sum(valids)

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                returns=returns,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
            )

        # Baseline fitting handled elsewheres...

        # logger.log("fitting baseline...")
        # if hasattr(self.algo.baseline, 'fit_with_samples'):
        #     self.algo.baseline.fit_with_samples(paths, samples_data)
        # else:
        #     self.algo.baseline.fit(paths)
        # logger.log("fitted")

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular_misc_stat('TrajLen', [len(p["rewards"]) for p in paths], placement='front')
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular_misc_stat('Return', undiscounted_returns, placement='front')

        return samples_data
