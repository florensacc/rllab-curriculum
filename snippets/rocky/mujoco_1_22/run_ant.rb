require_relative '../utils'

ctrl_cost_coeffs = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0]
impact_cost_coeffs = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0]
survive_rewards = [0, 1e-1, 1e-2, 1e-3, 1e-4]

shuffle_params(ctrl_cost_coeffs, impact_cost_coeffs, survive_rewards).take(1000).each do |ctrl_cost_coeff, impact_cost_coeff, survive_reward|

  params = {
    mdp: {
      _name: "mujoco_1_22.ant_mdp",
      ctrl_cost_coeff: ctrl_cost_coeff,
      impact_cost_coeff: impact_cost_coeff,
      survive_reward: survive_reward,
    },
    normalize_mdp: true,
    policy: {
      _name: "mean_std_nn_policy",
      hidden_sizes: [32, 32],
    },
    baseline: {
      _name: "linear_feature_baseline",
    },
    # exp_name: "ant_mujoco_1_22_ppo",
    algo: {
      _name: "trpo",
      whole_paths: true,
      batch_size: 50000,
      max_path_length: 500,
      n_itr: 100,
      #binary_search_penalty: false,
      # bs_kl_tolerance: 0.001,
      step_size: 0.01,
      # plot: true,
    },
    n_parallel: 4,
    snapshot_mode: "last",
    seed: 1,
    # plot: true,
  }
  #  command = to_docker_command(params)
  create_task_script(to_docker_command(params), launch: true, prefix: "ant")

  #puts command
  #system(command)
end
