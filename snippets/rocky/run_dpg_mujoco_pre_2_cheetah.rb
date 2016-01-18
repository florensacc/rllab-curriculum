require_relative './utils'

qf_learning_rates = [1e-4, 1e-5, 5e-5, 1e-6]
policy_learning_rates = [1e-4, 1e-5, 5e-5, 1e-6]
seeds = [1, 2, 3]

hidden_sizess = [[32, 32], [400, 300]]
qf_weight_decays = [0, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
soft_target_taus = [1e-4, 1e-3, 1e-2, 1e-1, 1]

shuffle_params(qf_learning_rates, policy_learning_rates, seeds, hidden_sizess, qf_weight_decays, soft_target_taus).each do |qf_learning_rate, policy_learning_rate, seed, hidden_sizes, qf_weight_decay, soft_target_tau|

  params = {
    mdp: {
      _name: "mujoco_pre_2.cheetah_mdp",
    },
    normalize_mdp: true,
    qf: {
      _name: "continuous_nn_q_function",
      hidden_sizes: hidden_sizes, # [32, 32],
    },
    policy: {
      _name: "mean_nn_policy",
      hidden_sizes: hidden_sizes, # [32, 32],
      hidden_nl: 'lasagne.nonlinearities.rectify',
      output_nl: 'lasagne.nonlinearities.tanh',
      output_W_init: 'lasagne.init.Uniform(-3e-3, 3e-3)',
      output_b_init: 'lasagne.init.Uniform(-3e-3, 3e-3)',
      bn: false,
    },
    algo: {
      _name: "dpg",
      batch_size: 64,
      n_epochs: 100,
      epoch_length: 10000,
      min_pool_size: 10000,
      replay_pool_size: 1000000,
      discount: 0.99,
      qf_weight_decay: qf_weight_decay,
      qf_learning_rate: qf_learning_rate,
      max_path_length: 100,
      eval_samples: 1,
      eval_whole_paths: true,
      soft_target_tau: soft_target_tau,
      policy_learning_rate: policy_learning_rate,
    },
    es: {
      _name: "ou_strategy",
    },
    snapshot_mode: "last",
    seed: seed,
  }
  # command = to_command(params)
  create_task_script(to_docker_command(params), launch: true, prefix: "dpg_cheetah")
  # puts command
  # system(command)

end
