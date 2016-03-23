require_relative '../../rocky/utils'

itrs = 500#100
batch_size = 50000
horizon = 500
discount = 0.99
seeds = (1..5).map do |i| i ** 2 * 5 + 23 end

mdps = []
# # loco
# mdps << "mujoco_1_22.swimmer_mdp"
# mdps << "mujoco_1_22.hopper_mdp"
mdps << "mujoco_1_22.walker2d_mdp"
mdps << "mujoco_1_22.half_cheetah_mdp"
mdps << "mujoco_1_22.ant_mdp"

load_map = {}
# load_map["mujoco_1_22.hopper_mdp"] = "loco_eval_61_43_mujoco_1_22.hopper_mdp_npg"
load_map["mujoco_1_22.walker2d_mdp"] = ['again_take3_bake_loco_15_68_mujoco_1_22.walker2d_mdp_npg', 'again_take3_bake_loco_11_28_mujoco_1_22.walker2d_mdp_npg', 'again_take3_bake_loco_17_103_mujoco_1_22.walker2d_mdp_npg', 'again_take3_bake_loco_19_148_mujoco_1_22.walker2d_mdp_npg', 'again_take3_bake_loco_13_43_mujoco_1_22.walker2d_mdp_npg']
load_map["mujoco_1_22.half_cheetah_mdp"] = ['again_take3_bake_loco_23_43_mujoco_1_22.half_cheetah_mdp_npg', 'again_take3_bake_loco_25_68_mujoco_1_22.half_cheetah_mdp_npg', 'again_take3_bake_loco_21_28_mujoco_1_22.half_cheetah_mdp_npg', 'again_take3_bake_loco_29_148_mujoco_1_22.half_cheetah_mdp_npg', 'again_take3_bake_loco_27_103_mujoco_1_22.half_cheetah_mdp_npg']
load_map["mujoco_1_22.ant_mdp"] = ['again_take3_bake_loco_37_103_mujoco_1_22.ant_mdp_npg', 'again_take3_bake_loco_35_68_mujoco_1_22.ant_mdp_npg', 'again_take3_bake_loco_39_148_mujoco_1_22.ant_mdp_npg', 'again_take3_bake_loco_31_28_mujoco_1_22.ant_mdp_npg', 'again_take3_bake_loco_33_43_mujoco_1_22.ant_mdp_npg']
load_map.clone.each do |k, vs|
    load_map[k] = vs.map do |v|
        "data/#{v}/params.pkl"
    end
end 

algos = []

# npg
[0.1].each do |ss|
    [1e0].each do |lr|
        algos << {
            _name: "npg",
            step_size: ss,
            update_method: "sgd",
            learning_rate: lr,
        }
    end
end
hss = []
hss << [100, 50, 25]

inc = 0
[true, false].each do |load|
    (load ? [true, false] : [true]).each do |train|
        mdps.each do |mdp|
            hss.each do |hidden_sizes|
                seeds.each do |seed|
                    algos.each do |algo|
                        exp_name = "again_take_hehefixedrandominit_bake_loco_#{inc = inc + 1}_#{seed}_#{mdp}_#{algo[:_name]}"
                        params = {
                            mdp: {
                                _name: mdp,
                            },
                            normalize_mdp: true,
                            policy: {
                                _name: "mean_std_nn_policy",
                                hidden_sizes: hidden_sizes,
                                load_params: load ? load_map[mdp].sample : nil,
                                load_params_masks: (haha=[true, true, true, true, false, false,
                                                          false, false, false]),
                                trainable_masks: train ? haha.map{|_| true} : haha.map(&:!),
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
                            }.merge(algo).merge(algo[:_name] == "cem" ? {} : {batch_size: batch_size}),
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
end
