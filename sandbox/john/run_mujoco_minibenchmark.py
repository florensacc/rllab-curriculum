import os
os.environ["THEANO_FLAGS"] = "device=cpu"
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.algos.ppo import PPO
from rllab.algos.trpo import TRPO
from rllab.misc.instrument import stub, run_experiment_lite, concretize
from rllab.envs.gym_env import GymEnv
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

stub(globals())
from rllab.misc.instrument import VariantGenerator

vg = VariantGenerator()
vg.add("envid", [
    "InvertedPendulum-v1", 
    "InvertedDoublePendulum-v1", 
    "Reacher-v1", 
    "HalfCheetah-v1", 
    "Swimmer-v1", 
    "Hopper-v1", 
    "Walker2d-v1", 
    "Ant-v1", 
    "Humanoid-v1", 
    "HumanoidStandup-v1"
    ])
vg.add("algo_cls", [TRPO, PPO])
vg.add("seed", [1])

for v in vg.variants()[0:1]:
    env = GymEnv(v['envid'], video_schedule=False)
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(8, 8)
    )
    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = v['algo_cls'](env=env, policy=policy, baseline=baseline, n_itr=10, batch_size=400)

    run_experiment_lite(
        algo.train(),
        exp_prefix="minibench",
        snapshot_mode="last",
        seed=v["seed"],
        mode="ec2",
        variant=v,
        n_parallel=0,
        dry=False
    )
