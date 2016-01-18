require_relative './utils'

qf_learning_rates = [1e-3, 1e-4, 1e-5, 1e-6]
policy_learning_rates = [1e-3, 1e-4, 1e-5, 1e-6]
seeds = [1, 2, 3]

shuffle_params(qf_learning_rates, policy_learning_rates, seeds).each do |qf_learning_rate, policy_learning_rate, seed|

  params = {
    mdp: {
      _name: "mujoco_pre_2.cheetah_mdp",
    },
    normalize_mdp: true,
    qf: {
      _name: "continuous_nn_q_function",
      hidden_sizes: [32, 32],
    },
    policy: {
      _name: "mean_nn_policy",
      hidden_sizes: [32, 32],
      hidden_nl: 'lasagne.nonlinearities.rectify',
      output_nl: 'lasagne.nonlinearities.tanh',
      output_W_init: 'lasagne.init.Uniform(-3e-3, 3e-3)',
      output_b_init: 'lasagne.init.Uniform(-3e-3, 3e-3)',
      bn: false,
    },
    algo: {
      _name: "dpg",
      batch_size: 32,
      n_epochs: 25,
      epoch_length: 10000,
      min_pool_size: 10000,
      replay_pool_size: 250000,
      discount: 0.99,
      qf_weight_decay: 0,
      qf_learning_rate: qf_learning_rate,
      max_path_length: 100,
      eval_samples: 1,
      eval_whole_paths: true,
      soft_target_tau: 0.001,
      policy_learning_rate: policy_learning_rate,
    },
    es: {
      _name: "ou_strategy",
    },
    snapshot_mode: "last",
    seed: seed,
  }
  command = to_command(params)
  puts command
  system(command)

end
