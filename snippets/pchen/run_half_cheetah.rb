require_relative '../rocky/utils'

quantile = 1
seed = 1

[1].each do |seed|
[0.01].each do |step_size|
params = {
  mdp: {
    _name: "box2d.half_cheetah_mdp",
    # trig_angle: false,
    # frame_skip: 2,
  },
  # normalize_mdp: nil,
  policy: {
    _name: "mean_std_nn_policy",
    # hidden_layers: [],
  },
  baseline: {
    _name: "linear_feature_baseline",
  },
  exp_name: "ppo_half_cheetah_seed_#{seed}_step_size_#{step_size}",
  algo: {
    _name: "ppo",
    binary_search_penalty: false,
    whole_paths: true,
    # quantile: quantile,
    batch_size: 10000,
    max_path_length: 1000,
    n_itr: 500,
    plot: true,
    step_size: step_size,

  },
  n_parallel: 4,
  # snapshot_mode: "none",
  seed: seed,
  plot: true,
}
command = to_command(params)
puts command
system(command)
end
end
