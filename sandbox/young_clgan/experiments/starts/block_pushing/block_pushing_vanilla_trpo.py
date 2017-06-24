
from rllab.misc import logger
from sandbox.young_clgan.logging import HTMLReport
from sandbox.young_clgan.logging import format_dict
from sandbox.young_clgan.logging.logger import ExperimentLogger
from sandbox.young_clgan.logging.visualization import plot_labeled_states


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

from sandbox.young_clgan.envs.block_pushing.pr2_env_lego_hand import Pr2EnvLego
from rllab import config
import os
import numpy as np
from sandbox.young_clgan.envs.start_env import generate_starts
from sandbox.young_clgan.envs.goal_start_env import GoalStartExplorationEnv


from sandbox.young_clgan.state.evaluator import label_states
from sandbox.young_clgan.envs.base import UniformListStateGenerator, UniformStateGenerator, FixedStateGenerator
from sandbox.young_clgan.state.generator import StateGAN
from sandbox.young_clgan.state.utils import StateCollection
# stub(globals())
#TODO: figure out crown goal generator

# seeds = [1, 33]
seeds = [1]

def run_task(v):
    # for inner environment, goal_generator shouldn't do anything and lego_generator shouldn't do anything
    goal_generator = PR2FixedGoalGenerator(goal = (0.6, 0.1, 0.5025)) # second dimension moves block further away vertically
    lego_generator = PR2LegoFixedBlockGenerator(block = (0.6, 0.2, 0.5025, 1, 0, 0, 0)) # want block at 0.6 +/- 0.2, , 0.1 +/- 0.4, 0.5025
    init_hand = np.array([0.6,  0.3,  0.5025])

    #for curriculum learning framework
    fixed_goal_generator = FixedStateGenerator(state=(0.6, 0.1, 0.5025))
    # TODO: make sure bounds are correct
    # uniform_start_generator = UniformStateGenerator(state_size=3, bounds=((-0.2, -0.4, 0), (0.2, 0.4, 0)),
                                                    # center=(0.6, 0.1, 0.5025))

    uniform_start_generator = UniformStateGenerator(state_size=3, bounds=((-0.05, -0.05, 0), (0.05, 0.05, 0)),
                                                    center=(0.6, 0.2, 0.5025))

    inner_env = normalize(Pr2EnvLego(
        goal_generator=goal_generator,
        lego_generator=lego_generator,
        # action_limiter=action_limiter,
        max_action=1,
        pos_normal_sample=True,
        qvel_init_std=0, #0.01,
        pos_normal_sample_std=.01, # ignored i think?
        fixed_target = init_hand, # sets the initial position of the hand to 0.6 0.3
        # use_depth=True,
        # use_vision=True,
        allow_random_restarts=True, #ignored i think?
    ))

    env = GoalStartExplorationEnv(
        env=inner_env,
        append_start=False,
        start_generator=uniform_start_generator,
        goal_generator=fixed_goal_generator,
        obs2goal_transform=lambda x: -1 * (x[-6:-3] - x[-3:]),   # TODO: check by setting breakpoint in goalenv
        # transform is -1 * [ (lego - goal) - lego] (final target position)
        obs2start_transform=lambda x: x[-3:], #TODO, make sure transforms are correct!
        # start is just the initial lego position
        terminal_eps = 0.03,  # TODO: potentially make smaller?
        distance_metric = 'L2',
        extend_distance_rew = False,  # I think this turns off L2 distance reward
        # distance_rew = True, # check, I think this checks the desired distance
        terminate_env = True,
    )

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have n hidden layers, each with k hidden units.
        hidden_sizes=(64, 64, 64),
        init_std=0.1,
        output_gain=0.1,
        )

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    for outer_iter in range(1, 10):
        logger.log("Outer itr # %i" % outer_iter)
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
            )

        algo.train()


vg = VariantGenerator()
vg.add('seed', [1])
# vg.add('initial_hand_distance', list(range(0, 5))) # how far hand is initialized

#exp_name = "exp4"
for vv in vg.variants():
    run_task(vv) # uncomment when debugging

    run_experiment_lite(
        # algo.train(),
        stub_method_call=run_task,
        use_gpu=False,
        variant=vv,
        # Number of parallel workers for sampling
        # n_parallel=32,
        n_parallel=8,
        snapshot_mode="last",
        seed=vv['seed'],
        mode="local",
        # mode="ec2",
        exp_prefix="hand_env15",
        # exp_name= "decaying-decaying-gamma" + str(t),
        # plot=True,
    )
