import os

from rllab.misc.instrument import VariantGenerator, variant

from sandbox.sandy.adversarial.fgsm import fgsm_perturbation
from sandbox.sandy.adversarial.io_util import init_output_file, \
        save_performance, save_video_file, init_all_output_file, save_performance_to_all
from sandbox.sandy.adversarial.shared import get_average_return, load_model
from sandbox.sandy.adversarial.vis import visualize_adversary
from sandbox.sandy.misc.util import get_time_stamp

adversary_algo = 'fgsm'
adversary_algo_param_names = ['norm', 'fgsm_eps']  # Levels of all_output_h5
game = 'space-invaders'
# Frameskip = 3, 44x44:
#params_file = '/home/shhuang/src/rllab-private/data/s3/' + \
#              'trpo-space/exp015/exp015_20170112_174121_353120_space_invaders/itr_499.pkl'
# Frameskip = 4, 44x44:
params_file = '/home/shhuang/src/rllab-private/data/s3/' + \
              'trpo-space/exp016/exp016_20170113_185323_813550_space_invaders/itr_499.pkl'

# Frameskip = 3, 84x84:
#params_file = '/home/shhuang/src/rllab-private/data/s3/' + \
#              'trpo-space/exp016/exp016_20170113_185321_146337_space_invaders/itr_0.pkl'
# Frameskip = 4, 84x84:
#params_file = '/home/shhuang/src/rllab-private/data/s3/' + \
#              'trpo-space/exp016/exp016_20170113_185322_438848_space_invaders/itr_0.pkl'
output_dir_base = '/home/shhuang/src/rllab-private/data/local/rollouts'
exp_index = os.path.basename(__file__).split('.')[0] # exp_xxx
output_dir = os.path.join(output_dir_base, exp_index, exp_index + "_" + get_time_stamp())

batch_size = 20000
seed = 1  # TODO: Try a few different seeds

# variant params ---------------------------------------
class VG(VariantGenerator):
    @variant
    def fgsm_eps(self):
        #return [0, 0.0000625, 0.000125, 0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016]
        return [0, 0.0000625]

    @variant
    def norm(self):
        #return ['l1', 'l2', 'l-inf']
        return ['l-inf', 'l2']

variants = VG().variants()

prefix = '-'.join([adversary_algo, game])
obs_min = -1
obs_max = 1

algo, env = load_model(params_file, batch_size)
all_output_h5 = init_all_output_file(output_dir, adversary_algo, adversary_algo_param_names, batch_size)

print("#Experiments: %d" % len(variants))
for v in variants:
    fgsm_eps = v['fgsm_eps']
    norm = v['norm']
    print("fgsm_eps:", fgsm_eps, "| norm:", norm)

    output_fname = "{exp_index}_{norm}_{eps}.h5".format(
        exp_index=exp_index,
        norm=norm,
        eps=str(fgsm_eps).replace('.', '-')
    )
    output_h5 = init_output_file(output_dir, prefix, 'fgsm', \
                                 {'eps': fgsm_eps, 'norm': norm}, \
                                 fname=output_fname)
    adversary_algo_params = {'fgsm_eps': fgsm_eps, 'norm': norm}

    # Set up adversary function
    if adversary_algo == 'fgsm':
        adv_params = dict(
                fgsm_eps=fgsm_eps,
                norm=norm,
                obs_min=obs_min,
                obs_max=obs_max,
                output_h5=output_h5
        )
        adversary_fn = lambda x: fgsm_perturbation(x, algo, **adv_params)
    else:
        raise NotImplementedError

    # Run policy rollouts with FGSM adversary for N trials, get average return
    env.set_adversary_fn(adversary_fn)
    avg_return_adversary, paths = get_average_return(algo, seed)
    save_performance(output_h5, avg_return_adversary, len(paths))

    output_prefix = output_fname.split('.')[0]
    video_file = visualize_adversary(output_h5, output_dir, output_prefix)
    save_video_file(output_h5, video_file)

    save_performance_to_all(all_output_h5, avg_return_adversary, adv_params, len(paths))
