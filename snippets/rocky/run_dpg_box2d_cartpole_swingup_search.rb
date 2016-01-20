require_relative './utils'

seeds = (1..10).to_a

hiddens = [[], [32], [32, 32]]
qf_weight_decays = [0, 1e-3, 1e-4, 1e-5]
policy_lrs = [1e-4, 5e-4, 1e-5]
discounts = [0.99, 0.999, 0.9999, 1]

while true
  params = {
    mdp: {
      _name: "box2d.cartpole_swingup_mdp",
    },
    normalize_mdp: true,
    qf: {
      _name: "continuous_nn_q_function",
    },
    policy: {
      _name: "mean_nn_policy",
      hidden_sizes: hiddens.sample,
      output_nl: 'lasagne.nonlinearities.tanh',
    },
    # exp_name: "dpg_box2d_cartpole_swingup",
    algo: {
      _name: "dpg",
      batch_size: 32,
      n_epochs: 500,
      epoch_length: 1000,
      min_pool_size: 10000,
      replay_pool_size: 100000,
      discount: discounts.sample,
      qf_weight_decay: qf_weight_decays.sample,
      max_path_length: 100,
      eval_samples: 10000,
      eval_whole_paths: true,
      renormalize_interval: 1000,
      # normalize_qval: false,
      policy_learning_rate: policy_lrs.sample,
    },
    es: {
      _name: "ou_strategy",
    },
    n_parallel: 4,
    snapshot_mode: "last",
    seed: seeds.sample,
  }

  command = to_command(params)
  puts command
  system(command)
end
