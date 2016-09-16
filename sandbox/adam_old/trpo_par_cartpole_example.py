# from rllab.algos.trpo import TRPO
# from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.adam_old.trpo_par import TRPO_par
from sandbox.adam_old.linear_feature_baseline_par import LinearFeatureBaseline_par
from rllab.envs.gym_env import GymEnv
# from rllab.envs.box2d.cartpole_env import CartpoleEnv
# from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.normalized_env import normalize
# from rllab.policies.gaussian_gru_policy import GaussianGRUPolicy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from timeit import default_timer as timer
import os
os.environ['MKL_NUM_THREADS'] = '1'
print('    MKL_NUM_THREADS=', os.getenv('MKL_NUM_THREADS'))
# start_time = timer()

# from rllab.core.network import ConvNetwork
# from sandbox.pchen.envs.atari import AtariEnv
# from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy


# env = normalize(CartpoleEnv())
# env = normalize(CartpoleSwingupEnv())
env = normalize(GymEnv("Hopper-v1", record_video=False))

# mdp = AtariEnv(
#     game='space_invaders',
#     obs_type='image',
# )

# network = ConvNetwork(
#     input_shape=mdp.spec.observation_space.shape,
#     output_dim=mdp.spec.action_space.n,
#     hidden_sizes=(20,),
#     conv_filters=(16, 16),
#     conv_filter_sizes=(4, 4),
#     conv_strides=(2, 2),
#     conv_pads=(0, 0),
# )
# policy = CategoricalMLPPolicy(
#     env_spec=mdp.spec,
#     prob_network=network,
# )


# policy = GaussianGRUPolicy(
#     env_spec=env.spec,
#     # The neural network policy should have two hidden layers, each with 32 hidden units.
#     hidden_sizes=(4,),
#     learn_std=False,
#     state_include_action=False
# )

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32,32)
)


baseline = LinearFeatureBaseline_par(env_spec=env.spec)

algo = TRPO_par(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=1000,
    n_itr=3,
    discount=0.99,
    step_size=0.01,
    n_proc=6,
    cpu_order=[i for i in range(32)] # CHANGE BASED ON YOUR COMPUTER! (put in order of physical cores)
)
algo.train()


# end_time = timer()

# print 'Total time: ', end_time - start_time