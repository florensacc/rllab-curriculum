require_relative '../../rocky/utils'

itrs = 500#100
batch_size = 10000
horizon = 250
discount = 0.99
seeds = (1..5).map do |i| i ** 2 * 5 + 23 end

mdps = []
# basics
mdps << "box2d.car_parking_mdp"

algos = []

# npg
[0.1, 0.05, 0.01].each do |ss|
  [1e0].each do |lr|
    algos << {
      _name: "npg",
      step_size: ss,
      update_method: "sgd",
      learning_rate: lr,
    }
  end
end

# vpg
[0.01, 0.005, 0.001].each do |lr|
  algos << {
    _name: "vpg",
    update_method: "adam",
    learning_rate: lr,
  }
end

hss = []
hss << [64, 32, 16]

ranges = [0, 0.1, 0.5, 1]

inc = 0
ranges.each do |range|
  hss.each do |hidden_sizes|
    seeds.each do |seed|
      mdps.each do |mdp|
        algos.each do |algo|
          exp_name = "cp_range_test_#{inc = inc + 1}_#{seed}_#{mdp}_#{algo[:_name]}"
          params = {
            mdp: {
              _name: mdp,
            },
            normalize_mdp: true,
            policy: {
              _name: "mean_std_nn_policy",
              hidden_sizes: hidden_sizes,
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
  --device /dev/nvidia0:/dev/nvidia0 \
  --device /dev/nvidiactl:/dev/nvidiactl \
  --device /dev/nvidia-uvm:/dev/nvidia-uvm \
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

end
