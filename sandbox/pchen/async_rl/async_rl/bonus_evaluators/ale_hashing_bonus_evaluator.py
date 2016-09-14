import numpy as np
import multiprocessing as mp
import copy

from rllab.misc import logger

from sandbox.pchen.async_rl.async_rl.hash.sim_hash import SimHash
from sandbox.pchen.async_rl.async_rl.bonus_evaluators.base import BonusEvaluator
from sandbox.pchen.async_rl.async_rl.utils.shareable import Shareable
from sandbox.pchen.async_rl.async_rl.utils.picklable import Picklable

class ALEHashingBonusEvaluator(BonusEvaluator, Shareable, Picklable):
    """
    Uses a hash function to store states counts. Then assign bonus reward to under-explored.
    Input states might be pre-processed versions of raw states.
    """
    def __init__(
            self,
            state_dim,
            state_preprocessor=None,
            hash=None,
            bonus_coeff=0.05,
            state_bonus_mode="1/sqrt(n_s)",
            log_prefix="",
            locked_stats=False,
        ):
        """
        locked_stats: whether to put a lock on the statistics to ensure atomic modifications
        """
        self.init_params = locals()
        self.init_params.pop("self")
        self.state_dim = state_dim
        if state_preprocessor is not None:
            assert state_preprocessor.output_dim == state_dim
            self.state_preprocessor = state_preprocessor
        else:
            self.state_preprocessor = None

        if hash is not None:
            assert(hash.item_dim == state_dim)
            self.hash = hash
        else:
            # Default: SimHash
            sim_hash_args = {
                "dim_key":64, "bucket_sizes":None
            }
            self.hash = SimHash(state_dim,**sim_hash_args)
        self.hash.reset()

        self.bonus_coeff = bonus_coeff
        self.state_bonus_mode = state_bonus_mode
        self.log_prefix = log_prefix
        self.locked_stats = locked_stats

        # logging stats ---------------------------------
        self.epoch_hash_count_list = []
        self.epoch_bonus_list = []
        self.new_state_count = 0
        self.total_state_count = 0

        self.unpicklable_list = ["shared_params","new_state_count","total_state_count"]

    def extract_shared_params(self):
        # changes are non-atomic
        shared_params = {
            "new_state_count": mp.RawValue('l', self.new_state_count),
            "total_state_count": mp.RawValue('l', self.total_state_count),
        }
        # changes are atomic
        if self.locked_stats:
            shared_params["new_state_count_obj"] = mp.Value('l', self.new_state_count)
            shared_params["total_state_count_obj"] = mp.Value('l', self.total_state_count)

        return shared_params

    def prepare_sharing(self):
        self.shared_params = self.extract_shared_params()
        self.set_shared_params(self.shared_params)
        if self.state_preprocessor is not None:
            self.state_preprocessor.prepare_sharing()
        self.hash.prepare_sharing()

    def set_shared_params(self,params):
        self.new_state_count = np.frombuffer(
            params["new_state_count"],
            dtype=int,
        )[0]

        self.total_state_count = np.frombuffer(
            params["total_state_count"],
            dtype=int,
        )[0]

    def process_copy(self):
        new = ALEHashingBonusEvaluator(**self.init_params)
        if self.state_preprocessor is not None:
            new.state_preprocessor = self.state_preprocessor.process_copy()
        new.hash = self.hash.process_copy()
        new.set_shared_params(self.shared_params)
        new.shared_params = self.shared_params
        return new


    def preprocess(self,states):
        if self.state_preprocessor is not None:
            processed_states = self.state_preprocessor.process(states)
        else:
            processed_states = states
        return processed_states

    def update_and_evaluate(self, states):
        # compute hash codes
        processed_states = self.preprocess(states)
        keys = self.hash.compute_keys(processed_states)

        # count new state
        prev_counts = self.hash.query_keys(keys)
        cur_new_state_count = list(prev_counts).count(0)

        # update the shared hash table
        self.hash.inc_keys(keys)
        counts = self.hash.query_keys(keys)

        if self.locked_stats:
            with self.shared_params["new_state_count_obj"].get_lock():
                self.shared_params["new_state_count_obj"].value += cur_new_state_count
                self.new_state_count = self.shared_params["new_state_count_obj"].value
            with self.shared_params["total_state_count_obj"].get_lock():
                self.shared_params["total_state_count_obj"].value += cur_new_state_count
                self.total_state_count = self.shared_params["total_state_count_obj"].value
        else:
            # lock-free
            self.new_state_count += cur_new_state_count
            self.total_state_count += cur_new_state_count

        # compute bonus
        bonuses = self.compute_bonus(counts)
        self.epoch_hash_count_list += list(counts)
        self.epoch_bonus_list += list(bonuses)
        return bonuses


    def compute_bonus(self,state_counts):
        if self.state_bonus_mode == "1/n_s":
            bonus = 1./np.maximum(1.,state_counts)
        elif self.state_bonus_mode == "1/sqrt(n_s)":
            bonus = 1./np.sqrt(np.maximum(1., state_counts))
        else:
            raise NotImplementedError
        bonus *= self.bonus_coeff
        return bonus

    def reset(self):
        self.hash.reset()

    def finish_epoch(self,epoch,log):
        # record counts
        if log:
            logger.record_tabular_misc_stat(
                self.log_prefix + "_ReplayedStateCount",
                self.epoch_hash_count_list
            )
            logger.record_tabular(
                self.log_prefix + "NewStateCount",
                self.new_state_count,
            )
            logger.record_tabular(
                self.log_prefix + "TotalStateCount",
                self.total_state_count,
            )

            # record bonus
            logger.record_tabular_misc_stat(
                self.log_prefix + "_BonusReward",
                self.epoch_bonus_list
            )

        self.epoch_hash_count_list = []
        self.epoch_bonus_list = []
        self.new_state_count = 0
        if self.locked_stats:
            with self.shared_params["new_state_count_obj"].get_lock():
                self.shared_params["new_state_count_obj"].value = 0
