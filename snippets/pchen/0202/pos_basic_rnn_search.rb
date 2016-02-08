require_relative '../../rocky/utils'

itrs = 100
batch_size = 50000
horizon = 100
discount = 0.99
seeds = (1..5).map do |i| i ** 2 * 5 + 23 end

mdps = []
# basics
mdps << "box2d.cartpole_mdp"
# mdps << "box2d.mountain_car_mdp"
# mdps << "box2d.cartpole_swingup_mdp"
mdps << "box2d.double_pendulum_mdp"
# # mdps << "box2d.car_parking_mdp"
# mdps << "mujoco_1_22.inverted_double_pendulum_mdp"
# 
# # loco
# mdps << "mujoco_1_22.swimmer_mdp"
# # mdps << "mujoco_1_22.hopper_mdp"
# # mdps << "mujoco_1_22.walker2d_mdp"
# # mdps << "mujoco_1_22.half_cheetah_mdp"
# mdps << "mujoco_1_22.ant_mdp"
# # mdps << "mujoco_1_22.simple_humanoid_mdp"
# # mdps << "mujoco_1_22.humanoid_mdp"

algos = []

# # cem
[0.1].each do |best_frac|
  [0.001, 0.005, 0.01, 0.1, 1].each do |extra_std|
    [itrs*0.5, ].each do |extra_decay_time|
      algos << {
        _name: "cem",
        # n_samples: (batch_size*1.0/horizon).to_i,
        best_frac: best_frac,
        extra_std: extra_std,
        extra_decay_time: extra_decay_time.to_i,
      }
    end
  end
end

[0.001, 0.01, 0.1, 1].each do |ss|
  algos << {
    _name: "recurrent.rreps",
    epsilon: ss,
  }
end

[50].each do |max_opt_itr|
  algos << {
    _name: "recurrent.rerwr",
    max_opt_itr: max_opt_itr,
    positive_adv: true,
  }
end

[1, 0.1, 0.05, 0.01, 0.005, 0.001].each do |ss|
  [1e0].each do |lr|
    algos << {
      _name: "recurrent.rnpg",
      step_size: ss,
      update_method: "sgd",
      learning_rate: lr,
      grad_clip: 10,
    }
  end
end

[1, 0.1, 0.05, 0.01, 0.005, 0.001].each do |lr|
  algos << {
    _name: "recurrent.rvpg",
    update_method: "adam",
    learning_rate: lr,
  }
end

[1, 0.1, 0.01, 0.001].each do |ss|
  algos << {
    _name: "recurrent.rppo",
    step_size: ss,
  }
end

[1, 0.1, 0.01, 0.001].each do |ss|
  algos << {
    _name: "recurrent.rtrpo",
    step_size: ss,
    backtrack_ratio: 0.8,
    max_backtracks: 15,
    subsample_factor: 0.1,
    grad_clip: 10,
  }
end

hss = []
hss << [100, 50, 25]

inc = 0
hss.each do |hidden_sizes|
  seeds.each do |seed|
    mdps.each do |mdp|
      algos.each do |algo|
        exp_name = "yetagain_pos_basic_rnn_#{inc = inc + 1}_#{seed}_#{mdp}_#{algo[:_name]}"
        params = {
          mdp: {
            _name: mdp,
            position_only: true,
          },
          normalize_mdp: true,
          policy: {
            _name: "mean_std_rnn_policy",
            grad_clip: 10,
            # _name: "mean_std_nn_policy",
            # hidden_sizes: hidden_sizes,
          },
          baseline: {
            _name: "linear_feature_baseline",
          },
          exp_name: exp_name,
          algo: {
            whole_paths: true,
            max_path_length: horizon,
            n_itr: itrs,
            discount: discount,
            # plot: true,
          }.merge(algo).merge({batch_size: batch_size}),
          n_parallel: 8,
          snapshot_mode: "last",
          seed: seed,
          # plot: true,
        }
        command = to_command(params)
        # puts command
        # system(command)
        # command = "LD_LIBRARY_PATH=/root/workspace/rllab/private/mujoco/binaries/1_22/linux #{command}"
  # --device /dev/nvidia0:/dev/nvidia0 \
  # --device /dev/nvidiactl:/dev/nvidiactl \
  # --device /dev/nvidia-uvm:/dev/nvidia-uvm \
        dockerified = """docker run \
  -v ~/.bash_history:/root/.bash_history \
  -v /slave/theano_cache_docker:/root/.theano \
  -v /slave/theanorc:/root/.theanorc \
  -v ~/.vim:/root/.vim \
  -v /slave/gitconfig:/root/.gitconfig \
  -v ~/.vimrc:/root/.vimrc \
  -v /slave/dockerfiles/ssh:/root/.ssh \
  -v /slave/jupyter:/root/.jupyter \
  -v /home/ubuntu/data:/root/workspace/data \
  -v /slave/workspace:/root/workspace \
  -v `pwd`/rllab:/root/workspace/rllab \
  --env LD_LIBRARY_PATH=/root/workspace/rllab/private/mujoco/binaries/1_22/linux:/usr/local/cuda/lib64 \
  dementrock/starcluster:0131 #{command}"""
        # puts dockerified
        # system(dockerified)
        fname = "#{exp_name}.sh"
        f = File.open(fname, "w")
        f.puts dockerified
        f.close
        system("chmod +x " + fname)
        system("qsub -V -b n -l mem_free=8G,h_vmem=14G -r y -cwd " + fname)
        # if mdp =~ /parking/ or mdp =~ /\.double/
        #     puts `~/qs.sh | grep #{exp_name} | /usr/local/sbin/kill.rb`
        # end
      end
    end
  end
end

