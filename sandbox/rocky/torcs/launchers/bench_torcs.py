from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.deterministic_mlp_policy import DeterministicMLPPolicy
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.algos.trpo import TRPO
from rllab.algos.vpg import VPG
from rllab.algos.tnpg import TNPG
from rllab.algos.ddpg import DDPG
from rllab.algos.ppo import PPO
from rllab.q_functions.continuous_mlp_q_function import ContinuousMLPQFunction
from rllab.exploration_strategies.ou_strategy import OUStrategy
from rllab.envs.normalized_env import normalize
from sandbox.rocky.torcs.envs.torcs_env import TorcsEnv

from rllab.misc.instrument import stub, run_experiment_lite
from rllab import config

stub(globals())

from rllab.misc.instrument import VariantGenerator, variant


class VG(VariantGenerator):
    @variant
    def seed(self):
        return [11, 21, 31, 41, 51]

    @variant
    def algo(self):
        # yield "ddpg"
        return ["trpo", "vpg", "ppo", "tnpg", "ddpg"]

    @variant
    def scale_reward(self, algo):
        if algo == "ddpg":
            return [1e-3, 1e-4, 1e-5, 1e-6]
        else:
            return [None]


vg = VG()

variants = vg.variants()
print("#Experiments:", len(variants))

for v in variants:
    env = normalize(TorcsEnv())
    policy = GaussianMLPPolicy(env_spec=env.spec, hidden_sizes=(128, 128))
    baseline = LinearFeatureBaseline(env_spec=env.spec)

    batch_args = dict(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=5000,
        max_path_length=5000,
        discount=0.99,
        n_itr=400,
    )
    if v["algo"] == "trpo":
        algo = TRPO(**batch_args)
    elif v["algo"] == "vpg":
        algo = VPG(**batch_args)
    elif v["algo"] == "tnpg":
        algo = TNPG(**batch_args)
    elif v["algo"] == "ppo":
        algo = PPO(**batch_args)
    elif v["algo"] == "ddpg":
        policy = DeterministicMLPPolicy(env_spec=env.spec, hidden_sizes=(128, 128))
        qf = ContinuousMLPQFunction(env_spec=env.spec, hidden_sizes=(128, 128))
        es = OUStrategy(env_spec=env.spec)
        algo = DDPG(
            env=env,
            policy=policy,
            qf=qf,
            es=es,
            n_epochs=200,
            epoch_length=10000,
            min_pool_size=1000,
            max_path_length=5000,
            eval_samples=5000,
            scale_reward=v["scale_reward"],
        )
    else:
        raise NotImplementedError

    config.KUBE_DEFAULT_RESOURCES = {
        "requests": {
            "cpu": 0.8,
        },
        "limits": {
            "cpu": 0.8,
        },
    }

    run_experiment_lite(
        algo.train(),
        n_parallel=1,
        seed=v["seed"],
        exp_prefix="bench_torcs_1",
        mode="lab_kube",
        snapshot_mode="last",
        variant=v,
    )
