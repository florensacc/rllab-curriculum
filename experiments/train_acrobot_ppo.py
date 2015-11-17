import os
os.environ['TENSORFUSE_MODE'] = 'theano'
os.environ['THEANO_FLAGS'] = 'device=cpu'
#import rllab.plotter as plotter
#plotter.init_worker()
from rllab.sampler import parallel_sampler
parallel_sampler.init_pool(4)
from rllab.vf.mujoco_value_function import MujocoValueFunction
from rllab.vf.no_value_function import NoValueFunction
from rllab.mdp.igor_mjc import AcrobotMDP
from rllab.algo.ppo import PPO
from rllab.policy.mujoco_policy import MujocoPolicy

if __name__ == '__main__':
    mdp = AcrobotMDP()
    policy = MujocoPolicy(mdp, hidden_sizes=[32, 32])
    vf = NoValueFunction()
    algo = PPO(
            exp_name='acrobot',
            samples_per_itr=4000,
            max_path_length=50,
            discount=0.99,
            stepsize=0.001,
            plot=False#True
    )
    algo.train(mdp, policy, vf)
