require_relative '../rocky/utils'

quantile = 1
seed = 1

params = {
  mdp: {
    # _name: "mujoco_1_22.half_cheetah_mdp",
    _name: "box2d.car_parking_mdp",
  },
  policy: {
    _name: "mean_std_nn_policy",
    # _name: "mean_nn_policy",
  },
  baseline: {
    _name: "linear_feature_baseline",
  },
  exp_name: "cem_ch_seed_#{seed}",
  algo: {
    # _name: "ppo",
    # step_size: 0.1,
    # binary_search_penalty: false,

    # _name: "trpo",
    # step_size: 0.2,
    # backtrack_ratio: 0.8,
    # max_backtracks: 10,
    # cg_iters: 10,
    # batch_size: 100,


    _name: "cem",
    batch_size: 50000,
    whole_paths: true,
    max_path_length: 500,
    n_itr: 500,
    plot: true,

  },
  n_parallel: 5,
  snapshot_mode: "last",
  seed: seed,
  plot: true,
}
command = to_command(params)
puts command
system(command)

