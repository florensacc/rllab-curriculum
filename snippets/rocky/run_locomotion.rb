require_relative './utils'

itrs = 1000
batch_size = 50000#0#0
horizon = 500
discount = 0.99
n_parallel = 4#10##1#2
seeds = (1..5).each do |i| i ** 2 * 5 + 23 end

mdps = []

mdps << {
  _name: "mujoco_1_22.ant_mdp",
}
mdps << {
  _name: "mujoco_1_22.half_cheetah_mdp",
}

#mdps << {
#  _name: "mujoco_1_22.humanoid_amputated_mdp",
#}
#
#mdps << {
#  _name: "mujoco_1_22.humanoid_mdp",
#}

mdps << {
  _name: "mujoco_1_22.walker2d_mdp",
}



#[0, 0.1, 1, 10].each do |alive_coeff|
#  [0, 0.1, 0.01, 0.001, 0.0001, 0.00001].each do |ctrl_cost_coeff|
    mdps << {
      _name: "mujoco_1_22.hopper_mdp",
      #ctrl_cost_coeff: ctrl_cost_coeff,
      #alive_coeff: alive_coeff,
    }
#  end
#end



#[0, 0.1, 0.01, 0.001, 0.0001, 0.00001].each do |ctrl_cost_coeff|
  mdps << {
    _name: "mujoco_1_22.swimmer_mdp",
    #ctrl_cost_coeff: ctrl_cost_coeff,
  }
#end

inc = 0
seeds.each do |seed|
  mdps.shuffle.each do |mdp|
    exp_name = "run_locomotion_#{inc = inc + 1}"
    params = {
      mdp: mdp,
      normalize_mdp: true,
      policy: {
        _name: "mean_std_nn_policy",
        hidden_sizes: [100, 50, 32],
      },
      baseline: {
        _name: "linear_feature_baseline",
      },
      exp_name: exp_name,
      algo: {
        _name: "parallel.trpo",
        batch_size: batch_size,
        whole_paths: true,
        max_path_length: horizon,
        n_itr: itrs,
        discount: discount,
        step_size: 0.1,
      },
      n_parallel: n_parallel,
      snapshot_mode: "last",
      seed: seed,
    }
    command = to_command(params)
    puts command
    system(command)
  end
end
