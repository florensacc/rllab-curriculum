import numpy as np
import multiprocessing as mp
import pickle
from rllab.misc import logger
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.sim_hash import SimHash
from sandbox.adam.parallel.util import SimpleContainer


class ALEActionHashingBonusEvaluator(object):
    """
    Uses a hash function to store state-action counts. Then assign bonus reward to under-explored.
    Input states might be pre-processed versions of raw states.
    """
    def __init__(
            self,
            state_dim,
            n_action,
            state_preprocessor=None,
            hash=None,
            bonus_form="1/sqrt(n)",
            log_prefix="",
            count_target="observations",
            parallel=False,
            retrieve_sample_size=np.inf,
            decay_within_path=False,
        ):
        self.state_dim = state_dim
        if state_preprocessor is not None:
            assert state_preprocessor.get_output_dim() == state_dim
            self.state_preprocessor = state_preprocessor
        else:
            self.state_preprocessor = None

        if hash is not None:
            assert(hash.item_dim == state_dim)
            self.hash = hash
        else:
            # Default: SimHash
            sim_hash_args = {
                "dim_key":64,
                "bucket_sizes":None,
                "parallel": parallel
            }
            self.hash = SimHash(state_dim,**sim_hash_args)
            self.hash.reset()
        self.n_action = n_action
        self.hash_list = [
            self.hash.get_copy()
            for i in range(n_action)
        ]
        # self.hash is for compute_keys()
        # self.hash_list for inc_keys() and query_keys()

        self.bonus_form = bonus_form
        self.log_prefix = log_prefix
        self.count_target = count_target
        self.parallel = parallel
        assert self.parallel == self.hash.parallel
        self.retrieve_sample_size = retrieve_sample_size
        self.decay_within_path = decay_within_path
        self.unpicklable_list = ["_par_objs","shared_dict"]
        self.snapshot_list = [""]

        # logging stats ---------------------------------
        self.rank = None


    def __getstate__(self):
        """ Do not pickle parallel objects. """
        state = dict()
        for k,v in iter(self.__dict__.items()):
            if k not in self.unpicklable_list:
                state[k] = v
            elif k in self.snapshot_list:
                state[k] = copy.deepcopy(v)
        return state

    # note that self.hash does not need parallelism
    def init_rank(self,rank):
        self.rank = rank
        for hash_a in self.hash_list:
            hash_a.init_rank(rank)

    def init_shared_dict(self, shared_dict):
        self.shared_dict = shared_dict
        for hash_a in self.hash_list:
            hash_a.init_shared_dict(shared_dict)

    def init_par_objs(self,n_parallel):
        n = n_parallel
        shareds = SimpleContainer(
            new_state_action_count_vec = np.frombuffer(
                mp.RawArray('l',n),
                dtype=int,
            ),
            total_state_action_count = np.frombuffer(
                mp.RawValue('l'),
                dtype=int,
            )[0],
            max_state_action_count_vec = np.frombuffer(
                mp.RawArray('l',n),
                dtype=int,
            ),
            min_state_action_count_vec = np.frombuffer(
                mp.RawArray('l',n),
                dtype=int,
            ),
            sum_state_action_count_vec = np.frombuffer(
                mp.RawArray('l',n),
                dtype=int,
            ),
            n_steps_vec = np.frombuffer(
                mp.RawArray('l',n),
                dtype=int,
            ),
        )
        barriers = SimpleContainer(
            summarize_count = mp.Barrier(n),
            update_count = mp.Barrier(n),
        )
        self._par_objs = (shareds, barriers)


    def preprocess(self,states):
        if self.state_preprocessor is not None:
            processed_states = self.state_preprocessor.process(states)
        else:
            processed_states = states
        return processed_states

    def retrieve_keys(self,paths):
        # do it path by path to avoid memory overflow
        keys = None
        for path in paths:
            path_len = len(path["rewards"])
            k = min(path_len, self.retrieve_sample_size)
            for i in range(0,path_len,k):
                if self.count_target == "observations":
                    states = path["observations"][i:i+k]
                else:
                    states = path["env_infos"][self.count_target][i:i+k]
                states = self.preprocess(states)
                new_keys = self.hash.compute_keys(states)
                if keys is None:
                    keys = new_keys
                else:
                    if isinstance(keys, np.ndarray):
                        keys = np.concatenate([keys,new_keys])
                    elif isinstance(keys, list):
                        keys = keys + new_keys
                    else:
                        raise NotImplementedError
        return keys

    def total_state_action_count(self):
        return sum(
            hash_a.total_state_count()
            for hash_a in self.hash_list
        )


    def fit_before_process_samples(self, paths):
        if self.parallel:
            shareds, barriers = self._par_objs

            # avoid re-computing keys and counts if they are already computed (probably in self.predict())
            # storing keys is necessary when converting the keys into memory and speed optimized forms takes time
            # storing counts is necessary if we count with a shared dict, which is slow in query
            if "keys" in paths[0]:
                example_path_keys = paths[0]["keys"]
                if isinstance(example_path_keys, list):
                    keys = []
                    for path in paths:
                        keys = keys + path["keys"]
                elif isinstance(example_path_keys, np.ndarray):
                    keys = np.concatenate([path["keys"] for path in paths])
                else:
                    raise NotImplementedError
            else:
                keys = self.retrieve_keys(paths)

            actions_onehot = np.vstack([path["actions"] for path in paths])
            actions = np.nonzero(actions_onehot)[1]

            # regroup keys by actions
            keys_list = [
                keys[np.where(actions==a)[0]]
                for a in range(self.n_action)
            ]

            if "counts" in paths[0]:
                prev_counts = np.concatenate([path["counts"] for path in paths])
            else:
                prev_counts = np.concatenate([
                    hash_a.query_keys(keys_a)
                    for hash_a, keys_a in zip(self.hash_list, keys_list)
                ])

            #FIXME: if a new state is encountered by more than one process, then it is counted more than once
            shareds.max_state_action_count_vec[self.rank] = max(prev_counts)
            shareds.min_state_action_count_vec[self.rank] = min(prev_counts)
            shareds.sum_state_action_count_vec[self.rank] = sum(prev_counts)
            shareds.n_steps_vec[self.rank] = len(prev_counts)

            barriers.summarize_count.wait()
            # avoid updating the hash table before we count new states

            if self.rank == 0:
                logger.record_tabular(
                    self.log_prefix + "StateActionCountMax",
                    max(shareds.max_state_action_count_vec),
                )
                logger.record_tabular(
                    self.log_prefix + "StateActionCountMin",
                    min(shareds.min_state_action_count_vec),
                )
                logger.record_tabular(
                    self.log_prefix + "StateActionCountAverage",
                    sum(shareds.sum_state_action_count_vec) / float(sum(shareds.n_steps_vec)),
                )
                prev_total_state_action_count = self.total_state_action_count()

            for hash_a, keys_a in zip(self.hash_list, keys_list):
                hash_a.inc_keys(keys_a)
            barriers.update_count.wait()

            if self.rank == 0:
                total_state_action_count = self.total_state_action_count()
                logger.record_tabular(
                    self.log_prefix + 'TotalStateActionCount',
                    total_state_action_count,
                )
                logger.record_tabular(
                    self.log_prefix + 'NewStateActionCount',
                    total_state_action_count - prev_total_state_action_count
                )
        else:
            keys = self.retrieve_keys(paths)
            actions_onehot = np.vstack([path["actions"] for path in paths])
            actions = np.nonzero(actions_onehot)[1]
            # regroup keys by actions
            keys_list = [
                keys[np.where(actions==a)[0]]
                for a in range(self.n_action)
            ]

            prev_counts = np.concatenate([
                hash_a.query_keys(keys_a)
                for hash_a, keys_a in zip(self.hash_list, keys_list)
            ])
            prev_total_state_action_count = self.hash.total_state_action_count()

            for hash_a, keys_a in zip(self.hash_list, keys_list):
                hash_a.inc_keys(keys_a)

            logger.record_tabular_misc_stat(self.log_prefix + 'StateActionCount',prev_counts)
            total_state_action_count = self.hash.total_state_action_count()
            logger.record_tabular(self.log_prefix + 'NewStateActionCount',total_state_action_count - prev_total_state_action_count)

            logger.record_tabular(
                self.log_prefix + 'TotalStateActionCount',
                total_state_action_count
            )


    def predict(self, path):
        keys = self.retrieve_keys([path])
        actions_onehot = path["actions"]
        actions = np.nonzero(actions_onehot)[1]
        keys_list = [
            keys[np.where(actions == a)[0]]
            for a in range(self.n_action)
        ]
        counts_misordered = np.concatenate([
            hash_a.query_keys(keys_a)
            for hash_a, keys_a in zip(self.hash_list, keys_list)
        ])  # not in the same order as states

        if self.decay_within_path:
            raise NotImplementedError
        else:
            counts = np.maximum(counts_misordered, 1)
        path["keys"] = keys
        path["counts"] = counts

        if self.bonus_form == "1/n":
            bonuses = 1./counts
        elif self.bonus_form == "1/sqrt(n)":
            bonuses = 1./np.sqrt(counts)
        elif self.bonus_form == "1/log(n+1)":
            bonuses = 1./np.log(counts + 1)
        else:
            raise NotImplementedError
        return bonuses

    def fit_after_process_samples(self, samples_data):
        pass

    def log_diagnostics(self, paths):
        pass
