import os
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'
os.environ['CUDA_VISIBLE_DEVICES']=''

import multiprocessing

# Symbols that need to be stubbed
import rllab
from rllab.misc.instrument import stub, run_experiment_lite

import matplotlib
matplotlib.use('Agg')

from sandbox.young_clgan.envs.point_env import PointEnv

EXPERIMENT_TYPE = 'cl_gan'


from sandbox.young_clgan.experiments.point_env_linear.cl_gan_algo import CLGANPointEnvLinear

stub(globals())

from sandbox.young_clgan.utils import AttrDict

if __name__ == '__main__':

    hyperparams = AttrDict(
        horizon=200,
        goal_size=2,
        goal_range=15,
        goal_noise_level=1,
        min_reward=5,
        max_reward=3000,
        improvement_threshold=10,
        outer_iters=200,
        inner_iters=5,
        pg_batch_size=20000,
        gan_outer_iters=5,
        gan_discriminator_iters=200,
        gan_generator_iters=5,
        gan_noise_size=4,
        gan_generator_layers=[256, 256],
        gan_discriminator_layers=[128, 128],
        experiment_type=EXPERIMENT_TYPE,
    )

    algo = CLGANPointEnvLinear(hyperparams)

    run_experiment_lite(
        algo.train(),
        pre_commands=['pip install --upgrade pip',
                      'pip install --upgrade theano',
                      'pip install --upgrade tensorflow',
                      'pip install tflearn',
                      'pip install dominate',
                      'pip install scikit-image',
                      ],
        n_parallel=multiprocessing.cpu_count(),
        use_cloudpickle=False,
        snapshot_mode="none",
        use_gpu=False,
    )
