require_relative '../rocky/utils'

quantile = 1
seed = 1

params = {
  mdp: {
    _name: "box2d.inverted_double_pendulum_mdp",
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_nn_policy",
    hidden_sizes: [],
  },
  baseline: {
    _name: "linear_feature_baseline",
  },
  exp_name: "trpo_idp_seed_#{seed}",
  algo: {
    _name: "ppo",
    step_size: 0.1,
    binary_search_penalty: false,
    batch_size: 5000,

    # _name: "trpo",
    # step_size: 1,
    # backtrack_ratio: 0.8,
    # max_backtracks: 10,
    # cg_iters: 10,
    # batch_size: 5000,


    # _name: "cem",

    # frame_skip: 2,
    whole_paths: true,
    max_path_length: 200,
    n_itr: 500,
    plot: true,

  },
  n_parallel: 4,
  snapshot_mode: "last",
  seed: seed,
  plot: true,
}
command = to_command(params)
puts command
system(command)

