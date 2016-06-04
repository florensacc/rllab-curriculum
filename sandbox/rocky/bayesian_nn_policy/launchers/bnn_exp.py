from __future__ import print_function
from __future__ import absolute_import

from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.box2d.cartpole_swingup_env import CartpoleSwingupEnv
from rllab.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from rllab.envs.mujoco.hopper_env import HopperEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from sandbox.rocky.bayesian_nn_policy.policies.bayesian_nn_policy import BayesianNNPolicy
from sandbox.rocky.bayesian_nn_policy.algos.bnn_algos import BNNTRPO, BNNPPO

from rllab.misc.instrument import stub, run_experiment_lite

stub(globals())

from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()
vg.add("env", map(normalize, [
    # CartpoleEnv(),
    # CartpoleSwingupEnv(),
    HalfCheetahEnv(),
    # HopperEnv()
]))

vg.add("seed", [11, 111, 211, 311, 411])

vg.add("std_mult", [1., 0.01, 1e-6])

vg.add("bbox_grad", ['all', 'none', 'std'])

vg.add("std_type", ['exp', 'identity', 'softplus'])

vg.add("batch_size", [10000])

variants = vg.variants()

print("#Experiments: %d" % len(variants))

for v in variants:
    policy = BayesianNNPolicy(
        GaussianMLPPolicy(env_spec=v["env"].spec),
        std_mult=v["std_mult"],
        std_type=v["std_type"],
        bbox_grad=v["bbox_grad"],
    )

    baseline = LinearFeatureBaseline(env_spec=v["env"].spec)

    algo = BNNTRPO(
        env=v["env"],
        policy=policy,
        baseline=baseline,
        max_path_length=500,
        batch_size=v["batch_size"],
        n_itr=500,
    )

    run_experiment_lite(
        algo.train(),
        exp_prefix="bnn_search_3",
        seed=v["seed"],
        n_parallel=1,
        snapshot_mode="last",
        mode="lab_kube"
    )
    # sys.exit()
