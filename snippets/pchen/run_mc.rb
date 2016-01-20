require_relative '../rocky/utils'

quantile = 1
seed = 1

params = {
  mdp: {
    _name: "box2d.mountain_car_mdp",
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
  exp_name: "ppo_mc_seed_#{seed}",
  algo: {
    _name: "ppo",
    binary_search_penalty: false,
    whole_paths: true,
    batch_size: 10000,
    max_path_length: 100,
    n_itr: 500,
    plot: true,
    step_size: 0.1,

  },
  n_parallel: 1,
  # snapshot_mode: "none",
  seed: seed,
  plot: true,
}
command = to_command(params)
puts command
system(command)

