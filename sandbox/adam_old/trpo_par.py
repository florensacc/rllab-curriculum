from rllab.algos.npo import NPO
# from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from rllab.core.serializable import Serializable

from sandbox.adam_old.conjugate_gradient_optimizer_par import ConjugateGradientOptimizer_par
from rllab.misc.overrides import overrides

import multiprocessing as mp
from multiprocessing.managers import BaseManager
from sandbox.adam.util import Barrier
import copy
from rllab.sampler import parallel_sampler
import numpy as np
import rllab.misc.logger as logger
from rllab.misc import special
from rllab.misc import tensor_utils
from rllab.algos import util
from rllab.misc import ext

from timeit import default_timer as timer
import cProfile

from sandbox.adam_old.util import TimingRecord
import psutil


class BarrierManager(BaseManager):
    pass


class TRPO_par(NPO, Serializable):
    """
    Trust Region Policy Optimization, Parallelized

    Notes for later:
    - Not sure what the Serializable class provides??
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            batch_size_init=100, # how small can this go without breaking anything?
            max_path_length_init=100,
            n_proc=1,
            cpu_order=None, # can be None or a list with order of processors assigned
            **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer_par(**optimizer_args)
        super(TRPO_par, self).__init__(optimizer=optimizer, optimizer_args=None, **kwargs)

        self.n_proc = n_proc
        self.batch_size_init = batch_size_init
        self.max_path_length_init = max_path_length_init
        self.batch_size = int(self.batch_size/n_proc) # To be collected by each process.
        self.total_batch_size = self.batch_size * n_proc
        self.whole_paths = False # Does not handle the 'True' case yet (bc pre-allocated shareds).
        self.cpu_order = cpu_order


    def define_timing(self):
        """
        ***The times recorded within each level of the algorithm must match this setup!***

        This is where the names of the times to be recorded are defined, as well as their
        hierarchical structure (i.e. which times are sub-components of other times).  

        Each of the 'keys_flat' dicts needs to have a '_main' key, denoting the top level
        dict.  Nested dicts are denoted by a list entry being equal to 'times_' appended to 
        the front of another key (arbitrary nesting levels possible).

        Entries in stamp_keys need to match the keys of the dict used to hold the timestamps
        (as written in the code where the timer function is called).  Layout of stamp_keys
        structure should match that of the keys_flat[dict_name] to be written to.
        """
        keys_flat = {}
        keys_flat['serial'] = {
            '_main': ['total', 'times_total'],
            'total': ['train_loop', 'init', 'times_init'],
            'init': ['serial_init', 'itr(theano)', 'init_vars'],
            }

        keys_flat['parallel'] = {
            'optimizer': ['loss_before', 'flat_g_w', 'flat_g_s',
                          'CG', 'step_size', 'bktrk'],
            'opt_pol': ['loss_before', 'wait_to_opt', 'optimizer',
                        'mean_kl', 'loss_after', 'times_optimizer'],
            'baseline': ['share','copy','fit'],
            '_main': ['sample', 'proc_samp', 'log_proc_samp',
                      'opt_pol', 'baseline', 'log_snapshot',
                      'times_opt_pol', 'times_baseline'],
            }

        to_share_dicts = {'serial': False, 'parallel': True}
        report_headers = {'serial': 'Serial (overall)',
                         'parallel': 'Parallel (Training Loop)'}

        stamp_keys = {}
        stamp_keys['train'] = {
            '_main': {'total': ['end','start'] },
            'total': {
                'init': ['init_vars', 'start'],
                'train_loop': ['end','init_vars'] 
                },
            'init': {
                'serial_init': ['serial_init','start'],
                'itr(theano)': ['init_iter', 'serial_init'],
                'init_vars': ['init_vars', 'init_iter'] 
                }
            }

        stamp_keys['train_loop'] = {
            '_main': {  'sample': ['sample', 'start'],
                        'proc_samp': ['proc_samp', 'sample'],
                        'log_proc_samp': ['log_proc_samp', 'proc_samp'],
                        'opt_pol': ['opt_pol', 'barrier_opt'],
                        'baseline': ['baseline_fit', 'opt_pol'],
                        'log_snapshot': ['log_snapshot', 'baseline_fit'],
                    },
            'baseline': {'share': ['baseline_share', 'opt_pol'] }
            }

        stamp_keys['opt_pol'] = {
            'opt_pol': {'loss_before': ['loss_before', 'start'],
                        'wait_to_opt': ['barrier_opt', 'loss_before'],
                        'optimizer': ['optimizer', 'barrier_opt'],
                        'mean_kl': ['mean_kl', 'optimizer'],
                        'loss_after': ['loss_after', 'mean_kl']
                        }
            }
 
        timing_defs = {
            'keys_flat': keys_flat,
            'to_share_dicts': to_share_dicts,
            'stamp_keys': stamp_keys,
            'report_headers': report_headers,
            'n_proc': self.n_proc
            }

        self.timing_defs = timing_defs        
    

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
        self.optimize_policy(-1, samples_data)
        self.policy.set_param_values(init_params) # un-do the little update
       

    def init_shared_vars(self):
        """
        To be called after init_iteration() and before spawning parallel processes.
        Later, shared variables will have '_s' suffix,
        while worker variables will have '_w' suffix.
        """
        n_proc = self.n_proc
        size_grad = np.size(self.policy.get_param_values(trainable=True))
        size_obs = self.env.spec.observation_space.flat_dim
        
        n_grad_elm_per_proc = -(-size_grad // n_proc) # (ceiling division)
        vb_idx = [n_grad_elm_per_proc * i for i in range(n_proc + 1)]
        vb_idx[-1] = size_grad # (using ceiling might overshoot the last)
        vb = [ (vb_idx[i], vb_idx[i+1]) for i in range(n_proc)]
        print vb

        # vec_block_factor = 3.5
        # n_vec_blocks = int(vec_block_factor * n_proc)
        # n_per_block = -(-size_grad // n_vec_blocks) # (ceiling division operator)
        # vb_indeces = [n_per_block * i for i in range(n_vec_blocks + 1)]
        # vb_indeces[-1] = size_grad
        # vb_bounds = [ (vb_indeces[i], vb_indeces[i+1]) for i in range(n_vec_blocks) ]
        # vb_starts = [int(vec_block_factor * proc) for proc in range(n_proc)]
        # vb_idx_lists = [ range(vb_starts[proc], n_vec_blocks) + range(vb_starts[proc]) for proc in range(n_proc)]

        proc_list = range(n_proc) # (rather than calling xrange() every time)

        par_data = {
            'n_proc': n_proc,
            'batch_size': self.batch_size, 
            'size_grad': size_grad, 
            'size_obs': size_obs, 
            'avg_fac': 1./self.n_proc, # (multiplies faster than divides?)
            # 'vb_bounds': vb_bounds,
            # 'vb_idx_lists': vb_idx_lists,
            'rank': None} # (rank filled in once inside the spawned process)

        # For optimize_policy_par():
        algo_shareds = {'loss_before': [mp.RawValue('d') for _ in proc_list], 
                        'mean_kl': [mp.RawValue('d') for _ in proc_list], 
                        'loss_after': [mp.RawValue('d') for _ in proc_list]}

        # For optimizer.optimize_par(): 
        all_grads_2d = np.reshape(np.frombuffer(mp.RawArray('d', size_grad * n_proc)), (size_grad,n_proc))
        grad = np.frombuffer(mp.RawArray('d', size_grad))
        # grad_np = np.frombuffer(grad)
        # flat_g_np = np.frombuffer(flat_g) # (will want numpy methods)
        # plain = mp.RawArray('d', size_grad)
        # plain_np = np.frombuffer((plain)) # (will want numpy methods)
        
        optimizer_shareds = {
            'loss_before': mp.RawValue('d'), 
            # 'flat_g': flat_g,
            # 'flat_g_np': flat_g_np, 
            # 'plain': plain, 
            # 'plain_np': plain_np,
            'all_grads_2d': all_grads_2d,
            'grad': grad,
            'vb_list': vb,
            'loss': mp.RawValue('d'), 
            'constraint_val': mp.RawValue('d'),
            'lock': mp.Lock(),
            # 'vec_locks': [mp.Lock() for _ in xrange(n_vec_blocks)]
        }

        # For baseline: 
        # Minimize memory usage by making a separate shared variable for each 
        # process: only one process will write to each one (so copies will not 
        # be made), and then the baseline-fitting process will access these in 
        # read-only manner.
        baseline_shareds = {
            'coeffs': mp.RawArray('d', np.size(self.baseline._coeffs)), # (only written-to by one)
            'observations': [mp.RawArray('d', self.batch_size * size_obs) for _ in proc_list], 
            'returns': [mp.RawArray('d', self.batch_size) for _ in proc_list],
            'start_indeces': [mp.RawArray('i', self.batch_size) for _ in proc_list], # (only need size=n_paths in each)
            'num_start_indeces': [mp.RawValue('i', self.batch_size) for _ in proc_list]}


        # For logging processed samples (LPS):
        LPS_shareds = {
            'average_discounted_return': mp.RawValue('d'),
            'entropy': mp.RawValue('d'),
            'n_paths': [mp.RawValue('i') for _ in proc_list],
            'undiscounted_returns': [mp.RawArray('d', self.batch_size) for _ in proc_list], # (only need size=num_paths in each)
            'baselines': [mp.RawArray('d', self.batch_size) for _ in proc_list],
            'returns': [mp.RawArray('d', self.batch_size) for _ in proc_list],
            'lock': mp.Lock()}

        shared_vars = {'algo': algo_shareds, 
                       'optimizer': optimizer_shareds, 
                       'baseline': baseline_shareds,
                       'LPS': LPS_shareds}


        BarrierManager.register('Barrier',Barrier)
        barrier_mgr = BarrierManager()
        barrier_mgr.start()
        
        # In principle, each barrier usage should be a distinct barrier.  But,
        # a single barrier should actually suffice for all uses in this code.
        # Somewhere in the middle: name them by  where they are used for easy
        # debugging.
        barrier_train = barrier_mgr.Barrier(self.n_proc)
        barrier_baseline = barrier_mgr.Barrier(self.n_proc)
        barrier_opt_pol = barrier_mgr.Barrier(self.n_proc)
        barrier_opt = barrier_mgr.Barrier(self.n_proc)
        barrier_Hx = barrier_mgr.Barrier(self.n_proc)
        barrier_bktrk = barrier_mgr.Barrier(self.n_proc)
        barrier_LPS = barrier_mgr.Barrier(self.n_proc)

        mgr_objs = {
            'barrier_train': barrier_train,
            'barrier_baseline': barrier_baseline,
            'barrier_opt_pol': barrier_opt_pol,
            'barrier_opt': barrier_opt,
            'barrier_Hx': barrier_Hx,
            'barrier_bktrk': barrier_bktrk,
            'barrier_LPS': barrier_LPS}

        return par_data, shared_vars, mgr_objs


    def set_affinity(self, rank, verbose=False):
        if self.cpu_order is not None:
            n_cpu = len(self.cpu_order)
            assigned_affinity = [self.cpu_order[rank % n_cpu]]
            proc = psutil.Process()
            proc.cpu_affinity(assigned_affinity)
            if verbose:
                affinities = '\nRank: {},  Affinity: {}'.format(rank, proc.cpu_affinity())
                print affinities


    def log_processed_samples_par(self, itr, samples_data, par_data, LPS_shareds, mgr_objs):
        """
        To be called by all processes simultaneously.
        'LPS' = log_processed_samples
        """
        rank = par_data['rank']
        avg_fac = par_data['avg_fac']

        average_discounted_return_s = LPS_shareds['average_discounted_return']
        entropy_s = LPS_shareds['entropy']
        n_paths_s = LPS_shareds['n_paths'][rank]
        undiscounted_returns_s = LPS_shareds['undiscounted_returns'][rank]
        baselines_s = LPS_shareds['baselines'][rank]
        returns_s = LPS_shareds['returns'][rank]
        LPS_lock = LPS_shareds['lock']

        barrier_LPS = mgr_objs['barrier_LPS']

        average_discounted_return_w = \
            np.mean([path["returns"][0] for path in samples_data['paths']])
        entropy_w = np.mean(self.policy.distribution.entropy(samples_data['agent_infos']))
        
        if rank == 0:
            average_discounted_return_s.value = 0.
            entropy_s.value = 0.
        barrier_LPS.wait()
        
        n_paths_w = len(samples_data['paths'])        
        undiscounted_returns_w = [sum(path["rewards"]) for path in samples_data['paths']]
        
        undiscounted_returns_s[:n_paths_w] = undiscounted_returns_w
        n_paths_s.value = n_paths_w
        # (Assumes all processes make the same number of samples:)
        baselines_s[:] = np.concatenate(samples_data['baselines']) 
        returns_s[:] = np.concatenate(samples_data['returns'])

        with LPS_lock:
            average_discounted_return_s.value += average_discounted_return_w * avg_fac
            entropy_s.value += entropy_w * avg_fac

        barrier_LPS.wait()
        if rank == 0:
            baselines_s_list = LPS_shareds['baselines']
            returns_s_list = LPS_shareds['returns']
            n_paths_s_list = LPS_shareds['n_paths']
            undiscounted_returns_s_list = LPS_shareds['undiscounted_returns']

            baselines_all = np.concatenate([base[:] for base in baselines_s_list])
            returns_all = np.concatenate([ret[:] for ret in returns_s_list])
            undiscounted_returns_all = np.concatenate( [und_ret[:n.value] 
                for und_ret, n in zip(undiscounted_returns_s_list, n_paths_s_list)] )

            ev = special.explained_variance_1d(baselines_all, returns_all)

            logger.record_tabular('Iteration', itr)
            logger.record_tabular('AverageDiscountedReturn',
                                  average_discounted_return_s.value)
            logger.record_tabular('AverageReturn', np.mean(undiscounted_returns_all))
            logger.record_tabular('ExplainedVariance', ev)
            logger.record_tabular('NumTrajs', sum([n.value for n in n_paths_s_list]))
            logger.record_tabular('Entropy', entropy_s.value)
            logger.record_tabular('Perplexity', np.exp(entropy_s.value))
            logger.record_tabular('StdReturn', np.std(undiscounted_returns_all))
            logger.record_tabular('MaxReturn', np.max(undiscounted_returns_all))
            logger.record_tabular('MinReturn', np.min(undiscounted_returns_all))

        barrier_LPS.wait()


    def process_samples_own(self, itr, paths):
        """
        To be call by each process independently.
        (Like the serial process_samples(), but without logging or baseline fitting.)
        """
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
                baselines=baselines, # (just for logging)
                returns=returns # (just for logging)
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

            average_discounted_return = \
                np.mean([path["returns"][0] for path in paths])

            samples_data = dict(
                observations=obs,
                actions=actions,
                advantages=adv,
                rewards=rewards,
                valids=valids,
                agent_infos=agent_infos,
                env_infos=env_infos,
                paths=paths,
                baselines=baselines, # (just for logging)
                returns=returns # (just for logging)
            )

        return samples_data


    def log_diagnostics_par(self, paths):
        """
        Since paths are not shared, this will require custom implementation by
        environment, policy, and baseline.  Shucks.
        """
        pass


    def _log_opt_pol(self, par_data, algo_shareds):
        """
        To be called by only one process.
        These values are not needed for computation, so just have one process
        collect them...ech not sure this is faster or makes it easier to read.
        """
        avg_fac = par_data['avg_fac']

        loss_before_s_list = algo_shareds['loss_before']
        mean_kl_s_list = algo_shareds['mean_kl']
        loss_after_s_list = algo_shareds['loss_after']

        loss_before = sum([l.value for l in loss_before_s_list]) * avg_fac
        mean_kl = sum([k.value for k in mean_kl_s_list]) * avg_fac
        loss_after = sum([l.value for l in loss_after_s_list]) * avg_fac

        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('MeanKL', mean_kl)
        logger.record_tabular('dLoss', loss_before - loss_after)


    def optimize_policy_par(self, itr, samples_data, par_data, shared_vars, mgr_objs):
        """
        To be called by all processes simultaneously.
        """
        rank = par_data['rank']

        loss_before_s = shared_vars['algo']['loss_before'][rank]
        mean_kl_s = shared_vars['algo']['mean_kl'][rank]
        loss_after_s = shared_vars['algo']['loss_after'][rank]

        barrier_opt_pol = mgr_objs['barrier_opt_pol']

        timestamps = {}

        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in self.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in self.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)
        if self.policy.recurrent:
            all_input_values += (samples_data["valids"],)
        
        timestamps['start'] = timer()
        
        loss_before_s.value = self.optimizer.loss(all_input_values)
        timestamps['loss_before'] = timer()
        
        # All processes enter this method together:
        barrier_opt_pol.wait() # Put this barrier here to make sure timing isn't misrepresented in optimizer.
        timestamps['barrier_opt'] = timer()
        times_optimizer = self.optimizer.optimize_par(itr, all_input_values, par_data, shared_vars['optimizer'], mgr_objs)
        timestamps['optimizer'] = timer()

        
        mean_kl_s.value = self.optimizer.constraint_val(all_input_values)
        timestamps['mean_kl'] = timer()
        loss_after_s.value = self.optimizer.loss(all_input_values)
        timestamps['loss_after'] = timer()
        # Barrier: wait for everyone to finish writing to their shared vars before logging.
        barrier_opt_pol.wait()
        if rank == 0:
            self._log_opt_pol(par_data, shared_vars['algo'])
        
        self.timing.record_timestamps(timestamps, dict_name='parallel', stamp_name='opt_pol')
        self.timing.record_time_dict(times_optimizer, dict_name='parallel', key_name='optimizer')

        
    @overrides
    def train(self):
        """
        Called externally, this method replaces serial train and spawns multiple processes
        for parallelized (single-agent) training.
        """
        self.define_timing() # (writes results to self.timing_defs)
        self.timing = TimingRecord(**self.timing_defs)
        
        timestamps = {}
        timestamps['start'] = timer()
        self.start_worker() # (Call parallel_sampler with n_parallel=1 (default))
        self.init_opt()
        timestamps['serial_init'] = timer()
        self.init_iteration() # (Force Theano compiles and prep other values)
        timestamps['init_iter'] = timer()
        par_data, shared_vars, mgr_objs = self.init_shared_vars()
        timestamps['init_vars'] = timer()
        processes = [ mp.Process(target=self.train_par_loop_profiled, 
                                args=(rank, par_data, shared_vars, mgr_objs)) 
                        for rank in range(self.n_proc) ]
        for p in processes: p.start()
        for p in processes: p.join()
        self.shutdown_worker()
        timestamps['end'] = timer()

        # (processes have already recorded 'parallel' timings.)
        self.timing.record_timestamps(timestamps, dict_name='serial', stamp_name='train')
        self.timing.print_report(['serial', 'parallel'])

    def train_par_loop_profiled(self, rank, par_data, shared_vars, mgr_objs, seed=0):
        cProfile.runctx('self.train_par_loop(rank, par_data, shared_vars, mgr_objs, seed)', globals(), locals(), 'prof%d.prof' %rank)   

    def train_par_loop(self, rank, par_data, shared_vars, mgr_objs, seed=0):
        """
        All processes run this in parallel, synchronized.
        """
        self.set_affinity(rank)
        
        par_data['rank'] = rank
        barrier_train = mgr_objs['barrier_train']
        barrier_baseline = mgr_objs['barrier_baseline']

        # Does the ext.set_seed need to be here?
        # ext.set_seed(seed + rank) 
        parallel_sampler.set_seed(seed + rank) # (or some other rank-dependent value)        

        timestamps = {}
        times_baseline = None
        for itr in xrange(self.start_itr, self.n_itr):
            with logger.prefix('itr #%d | ' % itr):
                
                timestamps['start'] = timer()

                # Processes collect and process samples independently:
                paths = self.obtain_samples(itr)
                timestamps['sample'] = timer() 
                samples_data = self.process_samples_own(itr, paths)
                timestamps['proc_samp'] = timer()

                # Logging (optional):
                barrier_train.wait()
                self.log_processed_samples_par(itr, samples_data, par_data, shared_vars['LPS'], mgr_objs)
                self.log_diagnostics_par(paths)
                timestamps['log_proc_samp'] = timer()
                barrier_train.wait() # Use barrier to make sure timing in opt_pol is not misrepresented.
                timestamps['barrier_opt'] = timer()
                
                # Processes enter the optimization together:
                self.optimize_policy_par(itr, samples_data, par_data, shared_vars, mgr_objs)                
                timestamps['opt_pol'] = timer()
                
                # Fit baseline for next iteration (could go anywhere after process_samples())
                # (This looks like a unit of code that could go into a separate function,
                # but might break it up to hide baseline fitting time.)
                self.baseline.share_samples(samples_data, par_data, shared_vars['baseline'])
                timestamps['baseline_share'] = timer()
                barrier_baseline.wait() # Wait for everyone to finish writing shared values.
                if rank == 0:  # (Have someone else do this during snapshot saving?)
                    times_baseline = self.baseline.fit_par(par_data, shared_vars['baseline'])
                # Barrier: wait for the new baseline coefficients to be written.
                barrier_baseline.wait()
                self.baseline.read_par(shared_vars['baseline'])
                timestamps['baseline_fit'] = timer()
                
                # Logging.
                if rank == 0:
                    logger.log("saving snapshot... after Optim!")
                    params = self.get_itr_snapshot(itr, samples_data)  # (npo.py: no shared vars)
                    if self.store_paths:
                        params["paths"] = samples_data["paths"] # (only paths from rank 0 saved!)
                    logger.save_itr_params(itr, params)
                    logger.log("saved")
                    logger.dump_tabular(with_prefix=False)
                    if self.plot:
                        self.update_plot()
                        if self.pause_for_plot:
                            raw_input("Plotting evaluation run: Press Enter to "
                                      "continue...")    
                timestamps['log_snapshot'] = timer()

                self.timing.record_timestamps(timestamps, dict_name='parallel', stamp_name='train_loop')


        # Write private timing info to shared timing dict.
        self.timing.write_to_shared(rank, dict_name='parallel')
  


