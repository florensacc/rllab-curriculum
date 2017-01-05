import argparse
import copy
import os
from rllab.algos.trpo import TRPO
from rllab.baselines.gaussian_conv_baseline import GaussianConvBaseline
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.policies.categorical_conv_policy import CategoricalConvPolicy
from sandbox.sandy.envs.atari_env import AtariEnv
from sandbox.sandy.train.network_args import icml_trpo_atari_args, nips_dqn_args, baseline_args

DATA_DIR = "/home/shhuang/src/rllab-private/data/local/experiment/"
PARAMS_FNAME = 'params.pkl'
BATCH_SIZE = 100000

stub(globals())

parser = argparse.ArgumentParser()
parser.add_argument('--resume_from_dir', type=str, default=None)
args = parser.parse_args()

env = normalize(AtariEnv('Pong-v3', force_reset=True))
network_args = nips_dqn_args  #icml_trpo_atari_args

policy = CategoricalConvPolicy(
    env_spec=env.spec,
    name='pong-policy',
    **network_args
)

network_args_for_vf = copy.deepcopy(network_args)
network_args_for_vf.pop("output_nonlinearity")
baseline = GaussianConvBaseline(
    env_spec=env.spec,
    regressor_args = dict(
    optimizer = ConjugateGradientOptimizer(
        subsample_factor = 0.1,
        cg_iters = 10,
        #name="vf_opt",
    ),
    use_trust_region=True,
    step_size=0.01,
    batchsize=BATCH_SIZE*10,
    normalize_inputs=True,
    normalize_outputs=True,
    **network_args_for_vf
    )
)

policy_opt_args = dict(
    cg_iters=10,
    reg_coeff=1e-3,
    subsample_factor=0.1,
    max_backtracks=15,
    backtrack_ratio=0.8,
    accept_violation=False,
    hvp_approach=None,
    num_slices=1, # reduces memory requirement
)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=BATCH_SIZE,
    max_path_length=4500,#env.horizon,
    n_itr=2000,
    discount=0.99,
    step_size=0.01,
    # Uncomment both lines (this and the plot parameter below) to enable plotting
    # plot=True,
    optimizer_args = policy_opt_args,
)

exp_args = dict(
    stub_method_call=algo.train(),
    # Number of parallel workers for sampling
    n_parallel=1,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies seed for experiment. If not provided, a random seed will be used
    seed=1234,
    # plot=True,
)

if args.resume_from_dir is not None:
    exp_args['resume_from'] = os.path.join(DATA_DIR, args.resume_from_dir,
                                           PARAMS_FNAME)

run_experiment_lite(**exp_args)
