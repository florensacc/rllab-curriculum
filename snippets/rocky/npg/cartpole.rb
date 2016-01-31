require_relative '../utils'

seed = 1

params = {
  mdp: {
    _name: "box2d.cartpole_mdp",
  },
  normalize_mdp: true,
  policy: {
    _name: "mean_std_nn_policy",
    hidden_sizes: [32, 32],
  },
  baseline: {
    _name: "linear_feature_baseline",
  },
  # exp_name: "ppo_box2d_cartpole_quantile_#{quantile}_seed_#{seed}",
  algo: {
    _name: "npg",
    #binary_search_penalty: false,
    whole_paths: true,
    batch_size: 1000,
    learning_rate: 0.1,
    max_path_length: 100,
    update_method: 'adam',
    n_itr: 100,
  },
  seed: seed,
}
command = to_command(params)
puts command
system(command)

