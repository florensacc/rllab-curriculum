import numpy as np
from rllab.algo.batch_polopt import BatchPolopt
from rllab.sampler import parallel_sampler
from rllab.misc import autoargs
from rllab.misc.special import explained_variance_1d, discount_cumsum
from rllab.misc.tensor_utils import pad_tensor
from rllab.algo.util import center_advantages
import rllab.misc.logger as logger
import rllab.plotter as plotter


class RecurrentBatchPolopt(BatchPolopt):
    """
    Base class for batch sampling-based policy optimization methods.
    This include various policy gradient methods like vpg, npg, ppo, trpo, etc.
    """

    @autoargs.inherit(BatchPolopt.__init__)
    def __init__(self, **kwargs):
        super(RecurrentBatchPolopt, self).__init__(**kwargs)

    def obtain_samples(self, itr, mdp, policy, baseline):
        samples_data = super(RecurrentBatchPolopt, self).obtain_samples(
            itr, mdp, policy, baseline)
        paths = samples_data["paths"]

        max_path_length = max([len(path["advantages"]) for path in paths])

        # make all paths the same length (pad extra advantages with 0)
        obs = [path["observations"] for path in paths]
        obs = [pad_tensor(ob, max_path_length, ob[0]) for ob in obs]

        if self.opt.center_adv:
            raw_adv = np.concatenate([path["advantages"] for path in paths])
            adv_mean = np.mean(raw_adv)
            adv_std = np.std(raw_adv) + 1e-8
            adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
        else:
            adv = [path["advantages"] for path in paths]
        adv = [pad_tensor(a, max_path_length, 0) for a in adv]

        actions = [path["actions"] for path in paths]
        actions = [pad_tensor(a, max_path_length, a[0]) for a in actions]
        pdists = [path["pdists"] for path in paths]
        pdists = [pad_tensor(p, max_path_length, p[0]) for p in pdists]

        valids = [np.ones_like(path["returns"]) for path in paths]
        valids = [pad_tensor(v, max_path_length, 0) for v in valids]

        samples_data["observations"] = np.asarray(obs)
        samples_data["advantages"] = np.asarray(adv)
        samples_data["actions"] = np.asarray(actions)
        samples_data["valids"] = np.asarray(valids)
        samples_data["pdists"] = np.asarray(pdists)

        return samples_data
