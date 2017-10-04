

import random
import numpy as np
import time
from rllab.envs.normalized_env import normalize
from rllab.misc import logger
from rllab.misc.instrument import stub, run_experiment_lite, VariantGenerator
from rllab.mujoco_py.mjlib import osp
from curriculum.envs.base import FixedStateGenerator
from curriculum.envs.goal_start_env import GoalStartExplorationEnv
from curriculum.envs.start_env import find_all_feasible_states_plotting, generate_starts, \
    parallel_check_feasibility, check_feasibility
from curriculum.logging import HTMLReport
from curriculum.envs.maze.maze_ant.ant_maze_start_env import AntMazeEnv


def benchmark_parallel(v):
    # test that prallel mapping actually speeds things up; helpful when >100 states
    fixed_goal_generator = FixedStateGenerator(state=v['ultimate_goal'])
    fixed_start_generator = FixedStateGenerator(state=v['ultimate_goal'])

    inner_env = normalize(AntMazeEnv())
    env = GoalStartExplorationEnv(
        env=inner_env,
        start_generator=fixed_start_generator,
        obs2start_transform=lambda x: x[:v['start_size']],
        goal_generator=fixed_goal_generator,
        obs2goal_transform=lambda x: x[-3:-1],
        terminal_eps=v['terminal_eps'],
        distance_metric=v['distance_metric'],
        extend_dist_rew=v['extend_dist_rew'],
        inner_weight=v['inner_weight'],
        goal_weight=v['goal_weight'],
        terminate_env=True,
    )
    for size in [100, 1000, 5000]:
        for parallel in [True, False]:
            starts = generate_starts(env, starts=[v['start_goal']], horizon=v['initial_brownian_horizon'],
                                     size=size,
                                     # size speeds up training a bit
                                     variance=v['brownian_variance'],
                                     # subsample=v['num_new_starts'],
                                     )
            start = time.time()
            if parallel:
                processes = -1
            else:
                processes = 1
            filtered_starts = parallel_check_feasibility(env=env, starts=starts, max_path_length=100, n_processes=processes)
            print("Size: {}  parallel: {}  time: {}".format(size, parallel, time.time() - start))

def run_task(v):
    random.seed(v['seed'])
    np.random.seed(v['seed'])
    logger.log("Initializing report...")
    log_dir = logger.get_snapshot_dir()
    if log_dir is None:
        log_dir = "/home/michael/"
    report = HTMLReport(osp.join(log_dir, 'report.html'), images_per_row=1)

    fixed_goal_generator = FixedStateGenerator(state=v['ultimate_goal'])
    fixed_start_generator = FixedStateGenerator(state=v['ultimate_goal'])

    inner_env = normalize(AntMazeEnv())
    env = GoalStartExplorationEnv(
        env=inner_env,
        start_generator=fixed_start_generator,
        obs2start_transform=lambda x: x[:v['start_size']],
        goal_generator=fixed_goal_generator,
        obs2goal_transform=lambda x: x[-3:-1],
        terminal_eps=v['terminal_eps'],
        distance_metric=v['distance_metric'],
        extend_dist_rew=v['extend_dist_rew'],
        inner_weight=v['inner_weight'],
        goal_weight=v['goal_weight'],
        terminate_env=True,
    )

    seed_starts = generate_starts(env, starts=[v['start_goal']], horizon=v['initial_brownian_horizon'], #TODO: increase to 2000
                                  size = 1000,
                                  # size speeds up training a bit
                                  variance=v['brownian_variance'],
                                  animated=True,
                                  subsample=v['num_new_starts'],

                                  )  # , animated=True, speedup=1)
    np.random.shuffle(seed_starts)

    logger.log("Prefilter seed starts: {}".format(len(seed_starts)))
    # breaks code
    # starts = seed_starts
    # # hack to not print code
    # seed_starts = [start for start in seed_starts if check_feasibility(start, env, 10)]
    # starts = np.array(starts)
    # seed_starts = starts
    logger.log("Filtered seed starts: {}".format(len(seed_starts)))

    # with env.set_kill_outside():
    feasible_states = find_all_feasible_states_plotting(env, seed_starts, report, distance_threshold=0.1, brownian_variance=1, size=8000,
                                                        animate=True,  limit = v['goal_range'],
                                                        check_feasible=True,
                                                        check_feasible_path_length=500,
                                  center = v['goal_center'])
    return

vg = VariantGenerator()
vg.add('seed', [2])
vg.add('maze_id', [0])  # default is 0
vg.add('terminal_eps', [0.3])
vg.add('start_size', [15])  # this is the ultimate start we care about: getting the pendulum upright
vg.add('start_goal', [[0, 4, 0.55, 1, 0, 0, 0, 0, 1, 0, -1, 0, -1, 0, 1,]])
# brownian params
vg.add('brownian_variance', [1])
vg.add('initial_brownian_horizon', [50])
vg.add('brownian_horizon', [200])
vg.add('ultimate_goal', lambda maze_id: [(0, 4)] if maze_id == 0 else [(2, 4), (0, 0)] if maze_id == 12 else [(4, 4)])
# goal-algo params
vg.add('min_reward', [0.1])
vg.add('max_reward', [0.9])
vg.add('distance_metric', ['L2'])
vg.add('extend_dist_rew', [False])
vg.add('inner_weight', [0])
vg.add('goal_weight', lambda inner_weight: [1000] if inner_weight > 0 else [1])
vg.add('regularize_starts', [0])
vg.add('num_new_starts', [200])
vg.add('goal_range',
           lambda maze_id: [4] if maze_id == 0 else [7])
vg.add('goal_center', lambda maze_id: [(2, 2)] if maze_id == 0 else [(0, 0)])
for vv in vg.variants():
    # run_experiment_lite(
    #     variant=vv,
    #     seed=vv['seed'],
    #     stub_method_call=benchmark_parallel,
    #     use_gpu=False,
    #     exp_prefix="ant_benchmark",
    #     mode="local",
    #     n_parallel=4,
    # )
    # run_task(vv)
    run_experiment_lite(
        variant=vv,
        seed=vv['seed'],
        stub_method_call=run_task,
        use_gpu=False,
        exp_prefix="ant_find_starts_best",
        mode="local",
        n_parallel=4,

    )

