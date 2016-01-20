require_relative '../utils'

params = {
  mdp: {
    _name: "mujoco_1_22.humanoid_mdp",
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_nn_policy",
    hidden_sizes: [300, 300],
  },
  baseline: {
    _name: "nn_baseline",
    #_name: "linear_feature_baseline",
    hidden_sizes: [],#300, 300],
    #max_opt_itr: 500,
  },
  exp_name: "humanoid",
  algo: {
    _name: "trpo",
    whole_paths: true,
    batch_size: 50000,
    max_path_length: 500,
    n_itr: 10000,
    step_size: 0.01,
  },
  n_parallel: 4,
  snapshot_mode: "last",
  seed: 1,
}
command = to_command(params)
puts command
system(command)
