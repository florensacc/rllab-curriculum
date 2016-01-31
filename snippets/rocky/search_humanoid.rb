require_relative './utils'


vel_deviation_cost_coeffs = [1, 10, 100, 0.1, 0.01, 0]
alive_bonuses = [0.2, 0.5, 1.0, 0.05, 0]
ctrl_cost_coeffs = [1e-2, 1e-3, 1e-1, 0]
impact_cost_coeffs = [1e-5, 1e-4, 1e-3, 0]
clip_impact_costs = [0.5, 0.15, 0.015, 0.05, 1]
step_sizes = [1, 0.1, 0.01]

seeds = (1..5).map { |i| i ** 2 * 5 + 23 }


shuffle_params(
  vel_deviation_cost_coeffs, alive_bonuses, ctrl_cost_coeffs,
  impact_cost_coeffs, clip_impact_costs, step_sizes, seeds
).first(1000).each do |vd, ab, cc, ic, ci, ss, s|
  params = {
    mdp: {
      _name: "mujoco_1_22.simple_humanoid_mdp",
      vel_deviation_cost_coeff: vd,
      alive_bonus: ab,
      ctrl_cost_coeff: cc,
      impact_cost_coeff: ic,
      clip_impact_cost: ci,
    },
    normalize_mdp: true,
    policy: {
      _name: "mean_std_nn_policy",
      hidden_sizes: [100, 50, 32],
    },
    baseline: {
      _name: "parallel.linear_feature_baseline",
    },
    exp_name: "simple_humanoid",
    algo: {
      _name: "parallel.trpo",
      whole_paths: true,
      batch_size: 100000,
      max_path_length: 1000,
      n_itr: 500,
      step_size: ss,
    },
    n_parallel: 4,
    snapshot_mode: "last",
    seed: s,
  }
  command = to_command(params)
  puts command
  system(command)
end
