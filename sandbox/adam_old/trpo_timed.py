from rllab.algos.npo import NPO
# from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from sandbox.adam.conjugate_gradient_optimizer_timed import ConjugateGradientOptimizer_timed
from rllab.core.serializable import Serializable
from rllab.misc.overrides import overrides
from timeit import default_timer as timer
from sandbox.adam.util import TimingRecord
import numpy as np
from rllab.misc import special, tensor_utils, ext
from rllab.algos import util
import rllab.misc.logger as logger
from rllab.sampler import parallel_sampler
import time


class TRPO_timed(NPO, Serializable):
    """
    Trust Region Policy Optimization with detailed timing.
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            batch_size_init=100,
            max_path_length_init=100,
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer_timed(**optimizer_args)
        super(TRPO_timed, self).__init__(optimizer=optimizer, **kwargs)

        self.batch_size_init = batch_size_init
        self.max_path_length_init = max_path_length_init
        # self.plot = False



    def define_timing(self):
        """
        The times recorded within each level of the algorithm must match this setup!
        """
        keys_flat = {}
        keys_flat['serial'] = {
            '_main': ['total', 'times_total'],
            'total': ['train_loop', 'init', 'times_init', 'times_train_loop'],
            'init': ['serial_init', 'itr(theano)'],
            'train_loop': ['sample', 'proc_samp', 'log_proc_samp',
                           'opt_pol', 'baseline', 'log_snapshot',
                           'times_opt_pol'],
            'opt_pol': ['loss_before', 'optimizer', 'mean_kl',
                        'loss_after', 'times_optimizer'],
            'optimizer': ['loss_before', 'flat_g', 'CG',
                          'step_size', 'bktrk']
        }

        to_share_dicts = {'serial': False}
        report_headers = {'serial': 'Overall'}

        stamp_keys = {}
        stamp_keys['train'] = {
            '_main': {'total': ['end', 'start'] },
            'total': {
                'init': ['init_iter', 'start'],
                'train_loop': ['end', 'init_iter']
            },
            'init': {
                'serial_init': ['serial_init','start'],
                'itr(theano)': ['init_iter', 'serial_init']
            }
        }
        stamp_keys['train_loop'] = {
            'train_loop': { 'sample': ['sample', 'start'],
                            'proc_samp': ['proc_samp', 'sample'],
                            'log_proc_samp': ['log_proc_samp', 'proc_samp'],
                            'opt_pol': ['opt_pol', 'log_proc_samp'],
                            'baseline': ['baseline','opt_pol'],
                            'log_snapshot': ['log_snapshot', 'baseline']
            }
        }
        stamp_keys['opt_pol'] = {
            'opt_pol': {'loss_before': ['loss_before', 'start'],
                        'optimizer': ['optimizer', 'loss_before'],
                        'mean_kl': ['mean_kl', 'optimizer'],
                        'loss_after': ['loss_after', 'mean_kl']
                        }
        }

        timing_defs = {
            'keys_flat': keys_flat,
            'to_share_dicts': to_share_dicts,
            'stamp_keys': stamp_keys,
            'report_headers': report_headers,
            'n_proc': 1
            }

        self.timing_defs = timing_defs  
 

    @overrides
    def process_samples(self, itr, paths):

        baselines = []
        returns = []
        for path in paths:
            path_baselines = np.append(self.baseline.predict(path), 0)
            deltas = path["rewards"] + \
                     self.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            path["returns"] = special.discount_cumsum(path["rewards"], self.discount)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        if not self.policy.recurrent:
            observations = tensor_utils.concat_tensor_list([path["observations"] for path in paths])
            actions = tensor_utils.concat_tensor_list([path["actions"] for path in paths])
            rewards = tensor_utils.concat_tensor_list([path["rewards"] for path in paths])
            advantages = tensor_utils.concat_tensor_list([path["advantages"] for path in paths])
            env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
            agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

            if self.center_adv:
                advantages = util.center_advantages(advantages)

            if self.positive_adv:
                advantages = util.shift_advantages_to_positive(advantages)



            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
                paths=paths,
                baselines=baselines,
                returns=returns
            )
        else:
            max_path_length = max([len(path["advantages"]) for path in paths])

            # make all paths the same length (pad extra advantages with 0)
            obs = [path["observations"] for path in paths]
            obs = np.array([tensor_utils.pad_tensor(ob, max_path_length) for ob in obs])

            if self.center_adv:
                raw_adv = np.concatenate([path["advantages"] for path in paths])
                adv_mean = np.mean(raw_adv)
                adv_std = np.std(raw_adv) + 1e-8
                adv = [(path["advantages"] - adv_mean) / adv_std for path in paths]
            else:
                adv = [path["advantages"] for path in paths]

            adv = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in adv])

            actions = [path["actions"] for path in paths]
            actions = np.array([tensor_utils.pad_tensor(a, max_path_length) for a in actions])

            rewards = [path["rewards"] for path in paths]
            rewards = np.array([tensor_utils.pad_tensor(r, max_path_length) for r in rewards])

            agent_infos = [path["agent_infos"] for path in paths]
            agent_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
            )

            env_infos = [path["env_infos"] for path in paths]
            env_infos = tensor_utils.stack_tensor_dict_list(
                [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
            )

            valids = [np.ones_like(path["returns"]) for path in paths]
            valids = np.array([tensor_utils.pad_tensor(v, max_path_length) for v in valids])

            

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
                baselines=baselines,
                returns=returns
            )

        # logger.log("fitting baseline...")
        # self.baseline.fit(paths)
        # logger.log("fitted")
        
        return samples_data


    def log_processed_samples(self, itr, samples_data):
        average_discounted_return = \
            np.mean([path["returns"][0] for path in samples_data['paths']])

        undiscounted_returns = [sum(path["rewards"]) for path in samples_data['paths']]

        ent = np.mean(self.policy.distribution.entropy(samples_data['agent_infos']))

        ev = special.explained_variance_1d(
            np.concatenate(samples_data['baselines']),
            np.concatenate(samples_data['returns'])
        )

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular('ExplainedVariance', ev)
        logger.record_tabular('NumTrajs', len(samples_data['paths']))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))



    @overrides
    def optimize_policy(self, itr, samples_data):
        timestamps = {}
        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        timestamps['start'] = timer()
        loss_before = self.optimizer.loss(all_input_values)
        timestamps['loss_before'] = timer()
        times_optimizer = self.optimizer.optimize(all_input_values)
        timestamps['optimizer'] = timer()
        mean_kl = self.optimizer.constraint_val(all_input_values)
        timestamps['mean_kl'] = timer()
        loss_after = self.optimizer.loss(all_input_values)
        timestamps['loss_after'] = timer()
        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)

        self.timing.record_timestamps(timestamps, dict_name='serial', stamp_name='opt_pol')
        self.timing.record_time_dict(times_optimizer, dict_name='serial', key_name='optimizer')



    def init_iteration(self):
        """
        To be called before spawning parallel processes.
        Performs short (serial) execution of obtain_samples(), process_samples(), and optimize_policy()
        to force theano compiles and prepare other values for dissemination to child processes.
        """
        init_params = self.policy.get_param_values()
        paths = parallel_sampler.sample_paths(
            policy_params=init_params,
            max_samples=self.batch_size_init, # (Reduced batch size.)
            max_path_length=self.max_path_length_init)
        samples_data = self.process_samples(-1, paths)
        import os
        os.environ['THEANO_FLAGS']='floatX=float32,device=gpu0'

        self.optimize_policy(-1, samples_data)
        self.policy.set_param_values(init_params) # un-do the little update


    @overrides
    def train(self):
    
        self.define_timing() # (writes results to self.timing_defs)
        self.timing = TimingRecord(**self.timing_defs)

        timestamps={}
        timestamps['start'] = timer()
        self.start_worker()
        self.init_opt()
        timestamps['serial_init'] = timer()
        self.init_iteration()
        timestamps['init_iter'] = timer()
        timestamps_loop = {}
        for itr in xrange(self.start_itr, self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                timestamps_loop['start'] = timer()
                paths = self.obtain_samples(itr)
                timestamps_loop['sample'] = timer()
                samples_data = self.process_samples(itr, paths)
                timestamps_loop['proc_samp'] = timer()
                self.log_processed_samples(itr, samples_data)
                self.log_diagnostics(paths)
                timestamps_loop['log_proc_samp'] = timer()
                self.optimize_policy(itr, samples_data)
                time.sleep(0.01)
                timestamps_loop['opt_pol'] = timer()
                self.baseline.fit(samples_data['paths'])
                timestamps_loop['baseline'] = timer()
                logger.log("saving snapshot... after Optim!")
                params = self.get_itr_snapshot(itr, samples_data)  # , **kwargs)
                if self.store_paths:
                    params["paths"] = samples_data["paths"]
                logger.save_itr_params(itr, params)
                logger.log("saved")
                logger.dump_tabular(with_prefix=False)
                if self.plot and False:
                    self.update_plot()
                    if self.pause_for_plot:
                        raw_input("Plotting evaluation run: Press Enter to "
                                  "continue...")
                timestamps_loop['log_snapshot'] = timer()
                self.timing.record_timestamps(timestamps_loop, dict_name='serial', stamp_name='train_loop')

        self.shutdown_worker()    
        timestamps['end'] = timer()
        self.timing.record_timestamps(timestamps, dict_name='serial', stamp_name='train')
        self.timing.print_report(['serial'])
