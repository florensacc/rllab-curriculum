from __future__ import print_function
from __future__ import absolute_import


from sandbox.rocky.nac.nac import NAC
from rllab.algos.trpo import TRPO
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.mujoco.swimmer_env import SwimmerEnv
from rllab.envs.normalized_env import normalize
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.misc.instrument import stub, run_experiment_lite
import lasagne.nonlinearities as NL
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
import joblib


stub(globals())

# data = joblib.load("data/local/nac/nac_2016_04_17_18_58_12_0001/params.pkl")


env = normalize(SwimmerEnv())#CartpoleEnv()
policy = GaussianMLPPolicy(env_spec=env.spec, output_nonlinearity=NL.tanh)
qf = ContinuousMLPQFunction(env_spec=env.spec, hidden_nonlinearity=NL.tanh)
baseline = LinearFeatureBaseline(env_spec=env.spec)

# algo = NAC(
#     env=env,#data["env"],
#     policy=policy,#data["policy"],
#     qf=qf,#data["qf"],
#     baseline=baseline,#data["baseline"],
#     qf_learning_rate=1e-3,
#     soft_target_tau=5e-3,
#     qf_update_itrs=10000,
#     eval_samples=1000,
#     policy_update_interval=10000,
#     policy_step_size=0.01,
#     max_path_length=500,
#     #scale_reward=0.1,
#     # policy_optimizer=PenaltyLbfgsOptimizer(),
# )

algo = TRPO(
    env=env,
    policy=policy,
    baseline=ZeroBaseline(env_spec=env.spec),
    batch_size=10000,
    # center_adv=False,
    max_path_length=500,
    # optimizer=ConjugateGradientOptimizer(subsample_factor=1),
    # plot=True
)

run_experiment_lite(
    algo.train(),
    exp_prefix="nac",
    snapshot_mode="last",
    n_parallel=4,
    # plot=True
)