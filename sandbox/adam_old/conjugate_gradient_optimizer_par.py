from rllab.misc import ext
from rllab.misc import krylov
from rllab.misc import logger
from rllab.core.serializable import Serializable
import theano.tensor as TT
import theano
import itertools
import numpy as np
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from timeit import default_timer as timer


class ConjugateGradientOptimizer_par(ConjugateGradientOptimizer):
    """
    Performs constrained optimization via line search. The search direction is computed using a conjugate gradient
    algorithm, which gives x = A^{-1}g, where A is a second order approximation of the constraint and g is the gradient
    of the loss function.
    """

    def __init__(
            self,
            cg_iters=10,
            reg_coeff=1e-5,
            subsample_factor=0.1,
            backtrack_ratio=0.8,
            max_backtracks=15,
            debug_nan=False):
        """

        :param cg_iters: The number of CG iterations used to calculate A^-1 g
        :param reg_coeff: A small value so that A -> A + reg*I
        :param subsample_factor: Subsampling factor to reduce samples when using "conjugate gradient. Since the
        computation time for the descent direction dominates, this can greatly reduce the overall computation time.
        :param debug_nan: if set to True, NanGuard will be added to the compilation, and ipdb will be invoked when
        nan is detected
        :return:
        """
        Serializable.quick_init(self, locals())
        self._cg_iters = cg_iters
        self._reg_coeff = reg_coeff
        self._subsample_factor = subsample_factor
        self._backtrack_ratio = backtrack_ratio
        self._max_backtracks = max_backtracks

        self._opt_fun = None
        self._target = None
        self._max_constraint_val = None
        self._constraint_name = None
        self._debug_nan = debug_nan
        


    def _print_debug_vals(self, name, rank, itr, sub_itr=None, short_vals=None, long_vals=None):
        """
        This was useful for debugging the parallelism.
        """
        report = ''
        if sub_itr is None:
            report += '\n[{}] {} debug values,  itr: {}\n'.format(rank, name, itr)
        else:
            report += '\n[{}] {} debug values,  itr: {},  sub_itr: {}\n'.format(rank, name, itr, sub_itr)
        if short_vals is not None:
            for k,v in short_vals.iteritems():
                report += '[{}] {}: {}\n'.format(rank, k, v)
        if long_vals is not None:
            for k,v in long_vals.iteritems():
                report += '[{}] {}[:3]: {}\n'.format(rank, k, v[:3]) 

        print report


    def optimize_par(self, itr, inputs, par_data, optimizer_shareds, mgr_objs, extra_inputs=None):
        """
        Processes execute this together in synchronized fashion.
        """
        rank = par_data['rank']
        avg_fac = par_data['avg_fac']
        # vb_bounds = par_data['vb_bounds']
        # vb_idx_list = par_data['vb_idx_lists'][rank]
        
        loss_before_s = optimizer_shareds['loss_before']
        all_grads_2d = optimizer_shareds['all_grads_2d']
        grad_s = optimizer_shareds['grad']
        vb = optimizer_shareds['vb_list'][rank]
        # flat_g_s = optimizer_shareds['flat_g']
        # flat_g_s_np = optimizer_shareds['flat_g_np']
        # plain_s = optimizer_shareds['plain']
        # plain_s_np = optimizer_shareds['plain_np']
        loss_s = optimizer_shareds['loss']
        constraint_val_s = optimizer_shareds['constraint_val']
        opt_lock = optimizer_shareds['lock']
        # vec_locks = optimizer_shareds['vec_locks']

        barrier_opt = mgr_objs['barrier_opt']
        barrier_Hx = mgr_objs['barrier_Hx']
        barrier_bktrk = mgr_objs['barrier_bktrk']

        inputs = tuple(inputs)
        if extra_inputs is None:
            extra_inputs = tuple()

        if self._subsample_factor < 1:
            n_samples = len(inputs[0])
            inds = np.random.choice(
                n_samples, n_samples * self._subsample_factor, replace=False)
            subsample_inputs = tuple([x[inds] for x in inputs])
        else:
            subsample_inputs = inputs

        if rank == 0:
            logger.log("computing loss before")

        stamp_start = timer()
        
        loss_before_w = self._opt_fun["f_loss"](*(inputs + extra_inputs))
        stamp_loss_before = timer()

        if rank == 0:
            logger.log("performing update")
            logger.log("computing descent direction")

        # flat_g_w = self._opt_fun["f_grad"](*(inputs + extra_inputs))
        all_grads_2d[:,rank] = self._opt_fun["f_grad"](*(inputs + extra_inputs))
        stamp_flat_g_w = timer()

        # theano.printing.debugprint(self._opt_fun['f_grad'])

        # Reset shared vars to 0 (used as accumulators).
        if rank == 0:
            loss_before_s.value = 0.
            # flat_g_s_np.fill(0.)

        # Barrier: wait for reset before adding to accumulators,
        # OR: wait for everyone to finish writing to grad_vec_s before adding.
        barrier_opt.wait()
        with opt_lock:
            loss_before_s.value += loss_before_w * avg_fac
            # flat_g_s[:] += flat_g_w * avg_fac

        # Everyone adds equal share of the grad elements (fastest)
        grad_s[vb[0]:vb[1]] = np.sum(all_grads_2d[vb[0]:vb[1],:], axis=1) * avg_fac


        # Shared vector blocking for increment (read+write) (fast)
        # for i in vb_idx_list:
        #     with vec_locks[i]:
        #         flat_g_s[vb_bounds[i][0]:vb_bounds[i][1]] += flat_g_w[vb_bounds[i][0]:vb_bounds[i][1]] * avg_fac

        # Barrier: wait for everyone to finish adding their contribution.
        barrier_opt.wait()
        stamp_flat_g_s = timer()

        # self._print_debug_vals('Flat_G', rank, itr, long_vals={'flat_g_w':flat_g_w, 'flat_g_s':flat_g_s})

        def Hx(x):
            """
            This function is parallelized, so uses of it need not be (e.g. krylov.cg() remains unchanged).
            """
            xs = tuple(self._target.flat_to_params(x, trainable=True))
            all_grads_2d[:,rank] = self._opt_fun["f_Hx_plain"](*(subsample_inputs + extra_inputs + xs)) + self._reg_coeff * x
            
             # Barrier: wait for everyone to be done reading from previously returned shared var before resetting.
            barrier_Hx.wait()
            # if rank == 0:
            #     plain_s_np.fill(0.)
            # Barrier: wait for reset before adding to accumulator.
            # barrier_Hx.wait()
            
            # Each process performs equal portion of additions. (fastest)
            grad_s[vb[0]:vb[1]] = np.sum(all_grads_2d[vb[0]:vb[1],:], axis=1) * avg_fac

            # # Shared vector blocking for increment (read+write) (medium)
            # for i in vb_idx_list:
            #     with vec_locks[i]:
            #         plain_s[vb_bounds[i][0]:vb_bounds[i][1]] += plain_w[vb_bounds[i][0]:vb_bounds[i][1]] * avg_fac
            
            # One writes at a time (slow)
            # with opt_lock:
            #     plain_s[:] += plain_w * avg_fac
            # Barrier: wait for everyone to finish adding their contribution.
            
            barrier_Hx.wait()

            # self._print_debug_vals('Plain', rank, itr, long_vals={'plain_w': plain_w, 'plain_s': plain_s})

            # WARNING: This returns the SHARED variable.  
            # (Avoids unnecessary copy: here and inside krylov.cg, it is only read from, not written to.) 
            return grad_s
            
        descent_direction = krylov.cg(Hx, grad_s, cg_iters=self._cg_iters, verbose=False)
        stamp_CG = timer()

        initial_step_size = np.sqrt(
            2.0 * self._max_constraint_val * (1. / (descent_direction.dot(Hx(descent_direction)) + 1e-8))
        )
        flat_descent_step = initial_step_size * descent_direction
        stamp_step = timer()

        # self._print_debug_vals('Descent', rank, itr, short_vals = {'initial_step_size': initial_step_size},
        #     long_vals={'descent_direction':descent_direction, 'flat_descent_step':flat_descent_step})

        if rank == 0:
            logger.log("descent direction computed")

        # (If desired, can try to split one of the processes off here to fit the 
        # baseline rather than participate in the backtracking.  Might require
        # a little juggling, but basically just need to share the final 'n_iter'
        # so the baseline-fitting process knows how to change its parameters.)

        prev_param = self._target.get_param_values(trainable=True)
        for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):
            cur_step = ratio * flat_descent_step
            cur_param = prev_param - cur_step
            self._target.set_param_values(cur_param, trainable=True)

            loss_w, constraint_val_w = self._opt_fun["f_loss_constraint"](*(inputs + extra_inputs))
            
            if rank == 0:
                loss_s.value = 0.
                constraint_val_s.value = 0.
            # Barrier: wait for reset before adding to accumulator.
            barrier_bktrk.wait()
            with opt_lock:
                loss_s.value += loss_w * avg_fac
                constraint_val_s.value += constraint_val_w * avg_fac
            # Barrier: wait for everyone to finish adding their contribution.
            barrier_bktrk.wait()

            # self._print_debug_vals('Backtrack', rank, itr, sub_itr=n_iter, 
            #     short_vals={'loss_w':loss_w, 'loss_s':loss_s.value, 'constraint_val_w': constraint_val_w,
            #     'constraint_val_s': constraint_val_s.value, 'loss_before_s': loss_before_s.value,
            #     '_max_constraint': self._max_constraint_val} )

            if self._debug_nan and np.isnan(constraint_val_s):
                import ipdb; ipdb.set_trace()
            if loss_s.value < loss_before_s.value and constraint_val_s.value <= self._max_constraint_val:
                break
            # Barrier: wait for everyone to check exit condition before repeating loop and reseting shareds.
            barrier_bktrk.wait()
        stamp_bktrk = timer()
        
        # self._print_debug_vals('Policy Param', rank, itr, short_vals={'n_bktrk_iter': n_iter},
        #     long_vals={'prev_param':prev_param, 'cur_param':cur_param})
        
        if rank == 0:
            logger.log("backtrack iters: %d" % n_iter)
            logger.log("computing loss after")
            logger.log("optimization finished")


        times = {
            'loss_before': stamp_loss_before - stamp_start,
            'flat_g_w': stamp_flat_g_w - stamp_loss_before,
            'flat_g_s': stamp_flat_g_s - stamp_flat_g_w,
            'CG': stamp_CG - stamp_flat_g_s,
            'step_size': stamp_step - stamp_CG,
            'bktrk': stamp_bktrk - stamp_step}

        return times

