require_relative '../rocky/utils'

quantile = 1
seed = 1

params = {
  mdp: {
    _name: "frozen_lake_mdp",
  },
  baseline: {
    _name: "linear_feature_baseline",
  },
  policy: {
    _name: "tabular_policy",
  },
  exp_name: "fl_npg",
  algo: {
    batch_size: 1000,

    _name: "npg",
    step_size: 0.01,
    update_method: 'sgd',
    learning_rate: 0.8,
    # uniform_state_dist: true,
  },
  n_parallel: 1,
  # snapshot_mode: "none",
  seed: seed,
}
command = to_command(params)
puts command
system(command)
