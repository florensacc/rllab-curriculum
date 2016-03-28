import os

os.environ['THEANO_FLAGS'] = 'device=cpu'
from sandbox.rocky.new_dpg import DPGExperiment
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize

# import sys

stub(globals())

# for seed in [1, 2, 3]:
#     for policy_lr in [1e-3, 1e-4, 1e-5]:
#         for qf_lr in [1e-3, 1e-4, 1e-5]:
#             for reward_scaling in [1, 0.01]:
#                 for qf_weight_decay in [0, 0.01]:
exp = DPGExperiment(
    policy_hidden_sizes=(32, 32),#400, 300),
    qf_hidden_sizes=(32, 32),#400, 300),
    policy_learning_rate=1e-4,
    qf_learning_rate=1e-3,
    reward_scaling=0.1,
    max_path_length=1000,
    qf_soft_target_tau=1e-3,
    policy_soft_target_tau=1e-3,
    qf_weight_decay=0,#.01,
    n_epochs=100,
)

env = normalize(CartpoleEnv())#, normalize_obs=True, normalize_reward=True)

run_experiment_lite(
    exp.run(env),
    # exp_prefix="new_dpg_search_cheetah_2",
    seed=0,
    # mode="ec2",
    # dry=True,
)
# sys.exit(0)
