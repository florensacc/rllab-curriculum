require_relative '../utils'

params = {
  mdp: {
    _name: "mujoco_1_22.swimmer_mdp",
  },
  policy: {
    _name: "mean_std_nn_policy",
  },
  algo: {
    _name: "cem",
    whole_paths: true,
    max_path_length: 100,
    n_itr: 500,
    plot: true,
  },
  n_parallel: 4,
  snapshot_mode: "last",
  plot: true,
}
command = to_command(params)
puts command
system(command)

