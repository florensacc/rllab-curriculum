require_relative '../utils'

# seed = 1
#
bns = [true]#, false]
qf_learning_rates = [1e-3]#, 1e-4, 1e-5, 5e-5, 1e-6]
policy_learning_rates = [1e-3]#1e-3, 1e-4, 1e-5, 5e-5, 1e-6]
seeds = [1, 12, 123, 1234, 12345]
qf_weight_decays = [1e-2]#0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
soft_target_taus = [5e-3]#1e-5, 1e-4, 1e-3, 1e-2]
discounts = [0.99]#, 0.95, 0.99, 0.999]
#ou_thetas = [0.15, 0.1, 0.05, 0.2, 0.25]

shuffle_params(
  bns, qf_learning_rates, policy_learning_rates, seeds, qf_weight_decays, soft_target_taus, discounts
).take(1000).each do |bn, qf_learning_rate, policy_learning_rate, seed, qf_weight_decay, soft_target_tau, discount|


  params = {
    mdp: {
      _name: "mujoco_1_22.half_cheetah_mdp",
    },
    normalize_mdp: true,
    qf: {
      _name: "continuous_nn_q_function",
      hidden_sizes: [100, 100],
      bn: bn,
    },
    policy: {
      _name: "mean_nn_policy",
      hidden_sizes: [400, 300],
      output_nl: 'lasagne.nonlinearities.tanh',
      bn: bn,
    },
    algo: {
      _name: "dpg",
      batch_size: 64,
      n_epochs: 20,
      epoch_length: 1000,
      min_pool_size: 1000,
      replay_pool_size: 1000000,
      discount: discount,
      qf_weight_decay: qf_weight_decay,#1e-2,#0,#1e-2,#0,#1e-3,
      qf_learning_rate: qf_learning_rate,#1e-4,
      max_path_length: 150,
      eval_samples: 10000,
      eval_whole_paths: true,
      soft_target: true,
      #hard_target_interval: 1000,
      soft_target_tau: soft_target_tau,#1e-3,
      policy_learning_rate: policy_learning_rate,#1e-4,
    },
    es: {
      _name: "ou_strategy",
      theta: 0.15,
      sigma: 0.3,
      #_name: "gaussian_strategy",
      #max_sigma: 0.5,#1.0,
      #min_sigma: 0.5,#0.1,
      #sigma_decay_range: 1000000,
    },
    n_parallel: 4,
    # seed: seed,
  }
  command = to_command(params)
  puts command
  system(command)
end
