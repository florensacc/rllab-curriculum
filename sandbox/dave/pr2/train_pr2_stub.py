import multiprocessing

from sandbox.dave.pr2.action_limiter import FixedActionLimiter
from sandbox.dave.rllab.algos.trpo import TRPO
# from rllab.algos.trpo import TRPO
from sandbox.dave.rllab.baselines.linear_feature_baseline import LinearFeatureBaseline

from rllab.envs.normalized_env import normalize
from sandbox.dave.rllab.goal_generators.goal_generator import FixedGoalGenerator
from sandbox.dave.rllab.goal_generators.pr2_goal_generators import PR2CrownGoalGeneratorSmall, PR2FixedGoalGenerator #PR2CrownGoalGeneratorSmall
from sandbox.dave.rllab.lego_generators.pr2_lego_generators import PR2LegoBoxBlockGeneratorSmall, PR2LegoBoxBlockGeneratorSmall,PR2LegoBoxBlockGeneratorSmall, PR2LegoFixedBlockGenerator
from rllab.misc.instrument import stub, run_experiment_lite
from sandbox.dave.rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
# from sandbox.dave.rllab.policies.gaussian_mlp_policy_tanh import GaussianMLPPolicy
from rllab.misc.instrument import VariantGenerator, variant


# from sandbox.dave.rllab.envs.mujoco.pr2_env_lego import Pr2EnvLego
# from sandbox.dave.rllab.envs.mujoco.pr2_env_lego_position import Pr2EnvLego
from sandbox.dave.rllab.envs.mujoco.pr2_env_lego_hand import Pr2EnvLego
# from sandbox.dave.rllab.envs.mujoco.pr2_env_reach import Pr2EnvLego
# from sandbox.dave.rllab.envs.mujoco.pr2_env_lego_position_different_objects import Pr2EnvLego
from rllab import config
import os
import numpy as np

# stub(globals())
#TODO: figure out crown goal generator

# seeds = [1, 33]
seeds = [1]

def run_task(v):
    # train_goal_generator = PR2CrownGoalGeneratorSmall()
    goal_generator = PR2FixedGoalGenerator(goal = (0.6, 0.1, 0.5025)) # second dimension moves block further away vertically
    lego_generator = PR2LegoFixedBlockGenerator(block = (0.6, 0.2, 0.5025, 1, 0, 0, 0))
    init_hand = np.array([0.6,  0.2 + v['initial_hand_distance'] * 0.05,  0.5025])

    # action_limiter = FixedActionLimiter()

    env = normalize(Pr2EnvLego(
        goal_generator=goal_generator,
        lego_generator=lego_generator,
        # action_limiter=action_limiter,
        max_action=1,
        pos_normal_sample=True,
        qvel_init_std=0, #0.01,
        pos_normal_sample_std=.01,  #0.5
        fixed_target = init_hand,
        # use_depth=True,
        # use_vision=True,
        allow_random_restarts=True,
        random_noise=v['random_noise'],
        # tip=t,
        # tau=t,
        # crop=True,
        # allow_random_vel_restarts=True,
    ))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have n hidden layers, each with k hidden units.
        hidden_sizes=(64, 64, 64),
        init_std=0.1,
        output_gain=0.1,
        # pkl_path= "data/local/train-Lego/IROS/3Dangle/exp_2/params.pkl"
        # json_path="/home/ignasi/GitRepos/rllab-goals/data/local/train-Lego/rand_init_angle_reward_shaping_continuex2_2016_10_17_12_48_20_0001/params.json",
        )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=5000,
        max_path_length=150,  #100
        n_itr=500, #50000
        discount=0.95,
        gae_lambda=0.98,
        step_size=0.01,
        # goal_generator=goal_generator,
        action_limiter=None,
        optimizer_args={'subsample_factor': 0.1},
        # discount_weights={'angle': 0.1, 'tip': .1},
        # plot=True,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        )

    algo.train()


# if __name__ == "__main__":
    # seed = 2
    # exp_name = "test_" + str(seed)

# exp2, just reward_tip reward
#exp3, fixed the goal
#exp4 fixed goal and target

vg = VariantGenerator()
vg.add('seed', [1])
# vg.add('initial_hand_distance', list(range(5, 10))) # how far hand is initialized
vg.add('initial_hand_distance', [3])
vg.add('random_noise', [True, False])
#exp_name = "exp4"
for vv in vg.variants():
    #run_task(vv)

    run_experiment_lite(
        # algo.train(),
        stub_method_call=run_task,
        use_gpu=False,
        variant=vv,
        # Number of parallel workers for sampling
        # n_parallel=32,
        n_parallel=8,
        # n_parallel=1,
        # pre_command   s=['pip install --upgrade pip',
        #               'pip install --upgrade theano'],
        # sync_s3_pkl=True,
        # periodic_sync=True,
        # sync_s3_png=True,
        # aws_config={"spot_price": '1.25', 'instance_type': 'm4.16xlarge'},
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        # seed=1,
        mode="local",
        # mode="ec2",
        # seed=s,
        # log_dir="data/local/train-Lego/trial_pretraining",
        # exp_prefix="train-Lego/RSS/trial",
        # exp_name="trial",
        # exp_prefix="train-Lego/RSS/torque-control",
        # exp_name="random_0001_param_torque_eve_fixed" + str(s),
        exp_prefix="hand_env24",
        # exp_name="dist_lessmass_noforcelimit",
        # exp_name= "decaying-decaying-gamma" + str(t),
        #exp_name=exp_name,
        # plot=True,
    )
