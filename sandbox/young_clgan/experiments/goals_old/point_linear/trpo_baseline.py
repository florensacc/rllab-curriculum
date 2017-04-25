import os
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''

import multiprocessing

import matplotlib
matplotlib.use('Agg')

from rllab.misc.instrument import stub, run_experiment_lite

EXPERIMENT_TYPE = 'trpo_baseline'

from sandbox.young_clgan.experiments.point_linear.trpo_baseline_algo import TRPOPointEnvLinear

stub(globals())

from sandbox.young_clgan.utils import AttrDict


if __name__ == '__main__':

    hyperparams = AttrDict(
        horizon=200,
        goal_size=2,
        goal_range=15,
        max_reward=6000,
        outer_iters=200,
        inner_iters=50,
        pg_batch_size=20000,
        experiment_type=EXPERIMENT_TYPE,
    )

    algo = TRPOPointEnvLinear(hyperparams)

    run_experiment_lite(
        algo.train(),
        n_parallel=multiprocessing.cpu_count(),
        use_cloudpickle=False,
        snapshot_mode="none",
        use_gpu=False,
    )
