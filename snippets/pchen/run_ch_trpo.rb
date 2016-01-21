require_relative '../rocky/utils'

quantile = 1
seed = 1

params = {
  mdp: {
    _name: "mujoco_1_22.half_cheetah_mdp",
    # trig_angle: false,
    # frame_skip: 2,
  },
  policy: {
    _name: "mean_std_nn_policy",
    # hidden_layers: [],
  },
  baseline: {
    _name: "linear_feature_baseline",
  },
  exp_name: "real_ppo_mc_seed_#{seed}",
  algo: {
    _name: "ppo",
    step_size: 0.01,
    binary_search_penalty: false,

    # _name: "trpo",
    # step_size: 0.01,
    # backtrack_ratio: 0.8,
    # max_backtracks: 10,
    # cg_iters: 10,

    whole_paths: true,
    batch_size: 10000,
    max_path_length: 100,
    n_itr: 500,
    plot: true,

  },
  n_parallel: 3,
  # snapshot_mode: "none",
  seed: seed,
  plot: true,
}
command = to_command(params)
puts command
system(command)

