from rllab.algo.base import RLAlgorithm

import theano.tensor as TT
import numpy as np

from rllab.misc import autoargs
from rllab.misc.special import discount_cumsum
from rllab.sampler import parallel_sampler
from rllab.sampler.parallel_sampler import pool_map, G
from rllab.sampler.utils import rollout
import rllab.misc.logger as logger
import rllab.plotter as plotter
import cma_es_lib


def sample_return(mdp, policy, params, max_path_length, discount):
    # mdp, policy, params, max_path_length, discount = args
    # of course we make the strong assumption that there is no race condition
    policy.set_param_values(params)
    path = rollout(
        mdp,
        policy,
        max_path_length,
    )
    path["returns"] = discount_cumsum(path["rewards"], discount)
    undiscounted_return = sum(path["rewards"])
    return dict(
        returns=path['returns'],
        undiscounted_return=undiscounted_return,
    )


class CEM(RLAlgorithm):

    @autoargs.arg("n_itr", type=int,
                  help="Number of iterations.")
    @autoargs.arg("max_path_length", type=int,
                  help="Maximum length of a single rollout.")
    @autoargs.arg("discount", type=float,
                  help="Discount.")
    @autoargs.arg("whole_paths", type=bool,
                  help="Make sure that the samples contain whole "
                       "trajectories, even if the actual batch size is "
                       "slightly larger than the specified batch_size.")
    @autoargs.arg("batch_size", type=int,
                  help="# of samples from trajs from param distribution.")
    @autoargs.arg("sigma0", type=float,
                  help="Initial std for param distribution.")
    @autoargs.arg("plot", type=bool,
                  help="Plot evaluation run after each iteration")
    def __init__(
            self,
            n_itr=500,
            max_path_length=500,
            discount=0.99,
            whole_paths=True,
            sigma0=1.,
            batch_size=None,
            extra_std=1.,
            extra_decay_time=100,
            plot=False,
            **kwargs
    ):
        super(CEM, self).__init__(**kwargs)
        self.plot = plot
        self.sigma0 = sigma0
        self.whole_paths = whole_paths
        self.discount = discount
        self.max_path_length = max_path_length
        self.n_itr = n_itr
        self.batch_size = batch_size

    def train(self, mdp, policy, **kwargs):

        cur_std = self.sigma0
        cur_mean = policy.get_param_values()
        es = cma_es_lib.CMAEvolutionStrategy(
            cur_mean, cur_std)

        parallel_sampler.populate_task(mdp, policy)
        if self.plot:
            plotter.init_plot(mdp, policy)

        cur_std = self.sigma0
        cur_mean = policy.get_param_values()

        itr = 0
        while itr < self.n_itr and not es.stop():

            if self.batch_size is None:
                # Sample from multivariate normal distribution.
                xs = es.ask()
                xs = np.asarray(xs)
                # For each sample, do a rollout.
                infos = (
                    pool_map(sample_return, [(x, self.max_path_length, self.discount) for x in xs]))
            else:
                cum_len = 0
                infos = []
                xss = []
                done = False
                while not done:
                    sbs = G.n_parallel * 2
                    # Sample from multivariate normal distribution.
                    # You want to ask for sbs samples here.
                    xs = es.ask(sbs)
                    xs = np.asarray(xs)

                    xss.append(xs)
                    sinfos = pool_map(
                        sample_return, [(x, self.max_path_length, self.discount) for x in xs])
                    for info in sinfos:
                        infos.append(info)
                        cum_len += len(info['returns'])
                        if cum_len >= self.batch_size:
                            xs = np.concatenate(xss)
                            done = True
                            break

            # Evaluate fitness of samples (negative as it is minimization
            # problem).
            fs = - np.array([info['returns'][0] for info in infos])
            # When batching, you could have generated too many samples compared
            # to the actual evaluations. So we cut it off in this case.
            xs = xs[:len(fs)]
            # Update CMA-ES params based on sample fitness.
            es.tell(xs, fs)

            logger.push_prefix('itr #%d | ' % itr)
            logger.record_tabular('Iteration', itr)
            logger.record_tabular('CurStdMean', np.mean(cur_std))
            undiscounted_returns = np.array(
                [info['undiscounted_return'] for info in infos])
            logger.record_tabular('AverageReturn',
                                  np.mean(undiscounted_returns))
            logger.record_tabular('StdReturn',
                                  np.mean(undiscounted_returns))
            logger.record_tabular('MaxReturn',
                                  np.max(undiscounted_returns))
            logger.record_tabular('MinReturn',
                                  np.min(undiscounted_returns))
            logger.record_tabular('AverageDiscountedReturn',
                                  np.mean(fs))
            logger.record_tabular('AvgTrajLen',
                                  np.mean([len(info['returns']) for info in infos]))

            logger.save_itr_params(itr, dict(
                itr=itr,
                policy=policy,
                mdp=mdp,
            ))
            logger.dump_tabular(with_prefix=False)
            if self.plot:
                plotter.update_plot(policy, self.max_path_length)
            # Update iteration.
            itr += 1

        # Set final params.
        policy.set_param_values(es.result()[0])
