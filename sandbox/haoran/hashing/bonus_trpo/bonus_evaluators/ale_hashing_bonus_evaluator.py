import numpy as np
import multiprocessing as mp
from rllab.misc import logger
from sandbox.haoran.hashing.bonus_trpo.bonus_evaluators.hash.sim_hash import SimHash
from sandbox.adam.parallel.util import SimpleContainer


class ALEHashingBonusEvaluator(object):
    """
    Uses a hash function to store states counts. Then assign bonus reward to under-explored.
    Input states might be pre-processed versions of raw states.
    """
    def __init__(
            self,
            state_dim,
            state_preprocessor=None,
            hash=None,
            bonus_form="1/sqrt(n)",
            log_prefix="",
            count_target="observations",
            parallel=False,
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

        self.bonus_form = bonus_form
        self.log_prefix = log_prefix
        self.count_target = count_target
        self.parallel = parallel
        assert self.parallel == self.hash.parallel

        # logging stats ---------------------------------
        self.epoch_hash_count_list = []
        self.epoch_bonus_list = []
        self.new_state_count = 0
        self.total_state_count = 0
        self.rank = None


    def __getstate__(self):
        """ Do not pickle parallel objects. """
        return {k: v for k, v in iter(self.__dict__.items()) if k != "_par_objs"}

    def init_rank(self,rank):
        self.rank = rank
        self.hash.init_rank(rank)

    def init_par_objs(self,n_parallel):
        n = n_parallel
        shareds = SimpleContainer(
            new_state_count_vec = np.frombuffer(
                mp.RawArray('l',n),
                dtype=int,
            ),
            total_state_count = np.frombuffer(
                mp.RawValue('l'),
                dtype=int,
            )[0],
        )
        barriers = SimpleContainer(
            new_state_count = mp.Barrier(n),
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
        if self.count_target == "observations":
            states = np.concatenate([p["observations"] for p in paths])
        else:
            states = np.concatenate([p["env_infos"][self.count_target] for p in paths])
        states = self.preprocess(states)
        keys = self.hash.compute_keys(states)
        return keys

    def fit_before_process_samples(self, paths):
        if self.parallel:
            shareds, barriers = self._par_objs
            keys = self.retrieve_keys(paths)
            prev_counts = self.hash.query_keys(keys)
            new_state_count = list(prev_counts).count(0)
            shareds.new_state_count_vec[self.rank] = new_state_count
            barriers.new_state_count.wait() # avoid updating the hash table before we count new states
            print("%d: before inc_keys"%(self.rank))
            self.hash.inc_keys(keys)
            print("%d: after inc_keys"%(self.rank))
            barriers.update_count.wait()

            if self.rank == 0:
                total_new_state_count = sum(shareds.new_state_count_vec)
                logger.record_tabular(
                    self.log_prefix + 'NewSteateCount',
                    total_new_state_count
                )
                shareds.total_state_count += total_new_state_count
                logger.record_tabular(
                    self.log_prefix + 'TotalStateCount',
                    shareds.total_state_count,
                )

        else:
            keys = self.retrieve_keys(paths)

            prev_counts = self.hash.query_keys(keys)
            new_state_count = list(prev_counts).count(0)

            self.hash.inc_keys(keys)
            counts = self.hash.query_keys(keys)

            logger.record_tabular_misc_stat(self.log_prefix + 'StateCount',counts)
            logger.record_tabular(self.log_prefix + 'NewSteateCount',new_state_count)

            self.total_state_count += new_state_count
            logger.record_tabular(self.log_prefix + 'TotalStateCount',self.total_state_count)

    def predict(self, path):
        keys = self.retrieve_keys([path])
        counts = self.hash.query_keys(keys)

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
