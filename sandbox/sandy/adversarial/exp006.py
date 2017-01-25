import os.path as osp

from rllab.misc.instrument import VariantGenerator, variant

from sandbox.sandy.adversarial.fgsm import fgsm_perturbation
from sandbox.sandy.adversarial.io_util import init_output_file, \
        save_performance, save_video_file, init_all_output_file, save_performance_to_all
from sandbox.sandy.adversarial.shared import get_average_return, load_model, load_models
from sandbox.sandy.adversarial.vis import visualize_adversary
from sandbox.sandy.misc.util import get_time_stamp

adversary_algo = 'fgsm'
adversary_algo_param_names = ['norm', 'fgsm_eps', 'policy_adv', 'policy_rollout']  # Levels of all_output_h5

experiments = ['async-rl_exp036']  # Format: algo-name_exp-index
games = ['chopper', 'pong', 'seaquest', 'space']

base_dir = '/home/shhuang/src/rllab-private/data/s3/'
output_dir_base = '/home/shhuang/src/rllab-private/data/local/rollouts'
exp_index = osp.basename(__file__).split('.')[0] # exp_xxx
output_dir = osp.join(output_dir_base, exp_index, exp_index + "_" + get_time_stamp())

batch_size = 20000
seed = 1  # TODO: Try a few different seeds
obs_min = -1
obs_max = 1

save_rollouts = False  # Set to True if you want to save and visualize rollouts
                       # Recommend setting this to False if there are a lot of runs,
                       # because saving rollouts takes up a *lot* of memory
test_transfer = False  # Whether to use adversarial examples generated on
                       # one model to attack a different model

use_gpu = True

if use_gpu:
    import os
    os.environ["THEANO_FLAGS"] = "device=gpu,dnn.enabled=auto,floatX=float32"
    import theano
    print("Theano config:", theano.config.device, theano.config.floatX)

policies = load_models(games, experiments, base_dir, batch_size, threshold=0.80, num_threshold=3)

# variant params ---------------------------------------
class VG(VariantGenerator):
    @variant
    def fgsm_eps(self):
        return [0.0000625, 0.000125, 0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008, 0]

    @variant
    def norm(self):
        return ['l1', 'l2', 'l-inf']

variants = VG().variants()

print("#Experiments: %d" % len(variants))
all_output_h5 = init_all_output_file(output_dir, adversary_algo,
                                     adversary_algo_param_names, batch_size)

for v in variants:
    fgsm_eps = v['fgsm_eps']
    norm = v['norm']
    print("fgsm_eps:", fgsm_eps, "| norm:", norm)

    for game in policies:
        game_policies = []
        for algo_name in policies[game]:
            game_policies += policies[game][algo_name]

        for policy_adv in game_policies:
            # Set up adversary function
            policy_adv[1].set_adversary_fn(None)
            for policy_rollout in game_policies:
                if not test_transfer and policy_rollout[3] != policy_adv[3]:
                    continue
                print(policy_adv[3], policy_rollout[3])
                output_h5 = None
                if save_rollouts:
                    output_fname = "{exp_index}_{norm}_{eps}_{policy_adv}_{policy_rollout}.h5".format(
                        exp_index=exp_index,
                        norm=norm,
                        eps=str(fgsm_eps).replace('.', '-'),
                        policy_adv=policy_adv[3],
                        policy_rollout=policy_rollout[3]
                    )
                    output_h5 = init_output_file(output_dir, None, 'fgsm', \
                                                 {'eps': fgsm_eps, 'norm': norm}, \
                                                 fname=output_fname)

                if adversary_algo == 'fgsm':
                    adv_params = dict(
                            fgsm_eps=fgsm_eps,
                            norm=norm,
                            obs_min=obs_min,
                            obs_max=obs_max,
                            output_h5=output_h5,
                            policy_adv=policy_adv[3],
                            policy_rollout=policy_rollout[3]
                    )
                    adversary_fn = lambda x: fgsm_perturbation(x, policy_adv[0], **adv_params)
                else:
                    raise NotImplementedError

                # Run policy rollouts with FGSM adversary for N trials, get average return
                policy_rollout[1].set_adversary_fn(adversary_fn)
                avg_return_adversary, paths = get_average_return(policy_rollout[0], seed)
                if save_rollouts:
                    save_performance(output_h5, avg_return_adversary, len(paths))

                    output_prefix = output_fname.split('.')[0]
                    video_file = visualize_adversary(output_h5, output_dir, output_prefix)
                    save_video_file(output_h5, video_file)

                print("Average Reward:", avg_return_adversary)
                save_performance_to_all(all_output_h5, avg_return_adversary, adv_params, len(paths))
            if fgsm_eps == 0:
                break
