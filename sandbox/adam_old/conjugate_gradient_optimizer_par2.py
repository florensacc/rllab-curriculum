from rllab.misc import ext
from rllab.misc import krylov
from rllab.misc import logger
from rllab.core.serializable import Serializable
import theano.tensor as TT
import theano
import itertools
import numpy as np
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
import time


# NEXT TO DO:
# RE-WRITE THIS WITH NO SUB-FUNCTION...it's not playing nice with multiprocessing.




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
        


    def optimize_par(self, itr, inputs, par_data, optimizer_shareds, mgr_objs,extra_inputs=None):
        rank = par_data['rank']
        n_proc = par_data['n_proc']
       
        loss_before_s = optimizer_shareds['loss_before']
        flat_g_s = optimizer_shareds['flat_g']
        plain_s = optimizer_shareds['plain']
        loss_s = optimizer_shareds['loss']
        constraint_val_s = optimizer_shareds['constraint_val']
        # opt_lock = optimizer_shareds['lock']

        barrier_opt = mgr_objs['barrier_opt']
        barrier_Hx = mgr_objs['barrier_Hx']
        barrier_bktrk = mgr_objs['barrier_bktrk']

        # nan_report = ''
        # nan_report += '\n NaN Report: [{}] BEGIN optimize_par, itr: {} \n'.format(rank, itr)
        # param_check = self._target.get_param_values(trainable=True)
        # nan_report += 'param_check: {} / {}\n'.format(np.sum(np.isnan(param_check)), np.size(param_check))
        # print nan_report

        # Make a view to the same shared data with numpy methods available.
        flat_g_np = np.frombuffer(flat_g_s.get_obj())
        plain_s_np = np.frombuffer(plain_s.get_obj())


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

        
        # Reset shared vars to 0 (used as accumulators).
        if rank == 0:
            loss_before_s.value = 0.
            flat_g_np.fill(0.)

        # Barrier
        barrier_opt.wait()

        if rank == 0:
            logger.log("computing loss before")

        # Average into a shared var:
        loss_before_w = self._opt_fun["f_loss"](*(inputs + extra_inputs))
        with loss_before_s.get_lock():
            loss_before_s.value += loss_before_w/n_proc
        
        if rank == 0:
            logger.log("performing update")
            logger.log("computing descent direction")


        # Average into a shared var:
        flat_g_w = self._opt_fun["f_grad"](*(inputs + extra_inputs))
        with flat_g_s.get_lock():
            flat_g_s[:] += flat_g_w/n_proc


        barrier_opt.wait()

        # Noticed that on one failure, the loops started with different p values.
        # Try to force a memory flush here:
        # flat_g_s[0] += 1e-12

        # flat_g = flat_g_np.copy()

        # g_report = ''
        # g_report += '\n Flat_G Report [{}], itr: {}\n'.format(rank, itr)
        # g_report += 'flat_g_w[:3]: {}\n'.format(flat_g_w[:3])
        # g_report += 'flat_g_s[:3]: {}\n'.format(flat_g_s[:3])
        # g_report += 'flat_g[:3]: {}\n'.format(flat_g[:3])
        # print g_report

        # loss_before_report = ''
        # loss_before_report += '\n Loss_Before Report [{}], itr: {}\n'.format(rank, itr)
        # loss_before_report += 'loss_before_w: {}\n'.format(loss_before_w)
        # loss_before_report += 'loss_before_s: {}\n'.format(loss_before_s.value)
        # print loss_before_report


        # KRYLOV.CG here:
        verbose = False
        residual_tol = 1e-10
        callback = None
        cg_iters = 10
        p = flat_g_np.copy()
        r = p.copy()
        x = np.zeros_like(r)
        rdotr = r.dot(r)

        fmtstr = "%10i %10.3g %10.3g"
        titlestr = "%10s %10s %10s"
        if verbose: print titlestr % ("iter", "residual norm", "soln norm")

        for i in xrange(cg_iters):
            
            # barrier_Hx.wait()

            if callback is not None:
                callback(x)
            if verbose: print fmtstr % (i, rdotr, np.linalg.norm(x))
            # z = f_Ax(p)
            # Old Hx here:
            xs = tuple(self._target.flat_to_params(p, trainable=True))
            
            plain_w = self._opt_fun["f_Hx_plain"](*(subsample_inputs + extra_inputs + xs)) + self._reg_coeff * p
            
            if rank == 0:
                plain_s_np.fill(0.)

            barrier_Hx.wait()

            with plain_s.get_lock():
                plain_s[:] += plain_w/n_proc

            barrier_Hx.wait()

            # Try to force a memory flush by writing (warning: requires synched shared if no lock)
            # plain_s[0] += 1e-12

            # z = plain_s_np.copy()

            
            # if (i == 0) or (i == cg_iters-1):
                # plain_report = ''
                # plain_report += '\n Plain Report [{}], itr: {}, cg_iter: {}'.format(rank, itr, i)
                # plain_report += '\n p[:3]: {}'.format(p[:3])
                # plain_report += '\n plain_w[:3]: {}'.format(plain_w[:3])
                # plain_report += '\n z[:3]: {}'.format(z[:3])
                # # plain_report += '\n plain[:3]: {}'.format(plain[:3])
                # print plain_report

            # END  Hx


            v = rdotr / (p.dot(plain_s) + 1e-8)
            x += v * p
            r -= v * plain_s
            newrdotr = r.dot(r)
            mu = newrdotr / (rdotr + 1e-8)
            p = r + mu * p

            rdotr = newrdotr

            barrier_Hx.wait()

            if rdotr < residual_tol:
                break


        if callback is not None:
            callback(x)
        if verbose: print fmtstr % (i + 1, rdotr, np.linalg.norm(x))  # pylint: disable=W0631
        
        descent_direction = x
        # END     Krylov.cg

        # Step size denominator
        # descent_direction.dot(Hx(descent_direction))
        # Old Hx:
        xs = tuple(self._target.flat_to_params(x, trainable=True))

        plain_w = self._opt_fun["f_Hx_plain"](*(subsample_inputs + extra_inputs + xs)) + self._reg_coeff * x

        if rank == 0:
            plain_s_np.fill(0.)

        barrier_Hx.wait()

        with plain_s.get_lock():
            plain_s[:] += plain_w/n_proc

        barrier_Hx.wait()
        # End old Hx




        # # I think we can get away with variables being in scope?
        # def Hx(x):
            
        #     xs = tuple(self._target.flat_to_params(x, trainable=True))
        #     #     rop = f_Hx_rop(*(inputs + xs))
            
        #     # Average into a shared var:
        #     plain_w = self._opt_fun["f_Hx_plain"](*(subsample_inputs + extra_inputs + xs)) + self._reg_coeff * x
            
        #     # Need a barrier here to make sure a process doesn't get ahead and
        #     # reset the shared var before some other process is done referencing it here
        #     # or in krylov.cg(), since this gets used in a loop.
        #     # (Alternative would be to make a private copy at the end here, and put the barrier
        #     # after that, but that's another copy we can avoid by doing it this way.)
        #     # barrier.wait() # well this didn't get rid of it.

        #     # # Reset shared variable for accumulation.
        #     # if rank == 0:
        #     #     plain_s_np.fill(0.)

        #     # # Barrier
        #     # barrier.wait()

        #     with plain_s.get_lock():
        #         plain_s[:] += plain_w/n_proc

        #     # Barrier
        #     barrier.wait()

        #     with plain_s.get_lock():
        #         plain = plain_s_np.copy()

        #     barrier.wait()

        #     if rank == 0:
        #         plain_s_np.fill(0.)

        #     # plain_report = ''
        #     # plain_report += '\n Plain Report [{}]'.format(rank)
        #     # plain_report += '\n x[:3]: {}'.format(x[:3])
        #     # plain_report += '\n plain_w[:3]: {}'.format(plain_w[:3])
        #     # plain_report += '\n plain_s[:3]: {}'.format(plain_s[:3])
        #     # print plain_report
        #     # assert np.allclose(rop, plain)
        #     # WARNING: This returns the SHARED variable.  But inside krylog.cg it is only read from, not written to. 
        #     return plain
        #     # alternatively we can do finite difference on flat_grad

        # # Barrier
        # barrier.wait()

        # # share_report = ''
        # # share_report += '\n Share Report: flat_g [{}]\n'.format(rank)
        # # share_report += 'flat_g_w[:3]: {}\n'.format(flat_g_w[:3])
        # # share_report += 'flat_g_s[:3]: {}\n'.format(flat_g_s[:3])
        # # print share_report

        # descent_direction = krylov.cg(Hx, flat_g_np, cg_iters=self._cg_iters, verbose=True)

        # print '\n[{}] descent_direction.dot(descent_direction): {}\n'.format(rank, descent_direction.dot(descent_direction))
        # print '\n[{}] descent_direction.dot(Hx(descent_direction)): {}\n'.format(rank, descent_direction.dot(Hx(descent_direction)))

        # initial_step_size = np.sqrt(
        #     2.0 * self._max_constraint_val * (1. / (descent_direction.dot(Hx(descent_direction)) + 1e-8))
        # )

        initial_step_size = np.sqrt(
            2.0 * self._max_constraint_val * (1. / (descent_direction.dot(plain_s) + 1e-8))
        )
        flat_descent_step = initial_step_size * descent_direction

        # descent_report = ''
        # descent_report += '\n Descent Report [{}], iter: {}\n'.format(rank, itr)
        # descent_report += 'descent_direction[:3]: {}\n'.format(descent_direction[:3])
        # descent_report += 'initial_step_size: {}\n'.format(initial_step_size)
        # descent_report += 'flat_descent_step[:3]: {}\n'.format(flat_descent_step[:3])
        # print descent_report

        
        # nan_report = ''
        # nan_report += '\n NaN Report: [{}] MIDDLE optimize_par\n'.format(rank)
        # nan_report += 'descent_direction: {} / {}\n'.format(np.sum(np.isnan(descent_direction)), np.size(descent_direction))
        # nan_report += 'initial_step_size: {} / {}\n'.format(np.sum(np.isnan(initial_step_size)), np.size(initial_step_size))
        # nan_report += 'flat_descent_step: {} / {}\n'.format(np.sum(np.isnan(flat_descent_step)), np.size(flat_descent_step))
        # print nan_report

        if rank == 0:
            logger.log("descent direction computed")

        # If desired, can try to split one of the processes off here to fit the 
        # baseline rather than participate in the backtracking.  Might require
        # a little juggling, but basically just need to share the final 'n_iter'
        # so the baseline process knows how to change its parameters.
        # nan_report = ''
        # nan_report += '\n NaN Report: [{}] Backtracking Line Search\n'.format(rank)

        prev_param = self._target.get_param_values(trainable=True)
        for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):
            cur_step = ratio * flat_descent_step
            cur_param = prev_param - cur_step
            self._target.set_param_values(cur_param, trainable=True)

            # nan_report += 'cur_step, {}: {} / {}\n'.format(n_iter, np.sum(np.isnan(cur_step)), np.size(cur_step))
            
            # Average into a shared var:
            loss_w, constraint_val_w = self._opt_fun["f_loss_constraint"](*(inputs + extra_inputs))
            
            if rank == 0:
                loss_s.value = 0.
                constraint_val_s.value = 0.

            # Barrier
            barrier_bktrk.wait()

            with loss_s.get_lock():
                loss_s.value += loss_w/n_proc

            with constraint_val_s.get_lock():
                constraint_val_s.value += constraint_val_w/n_proc
            
            # Barrier
            barrier_bktrk.wait()

            # bktrk_report = ''
            # bktrk_report += '\nBackTrack Report [{}], itr: {},  bktrk_itr: {}\n'.format(rank, itr, n_iter)
            # bktrk_report += 'loss_w: {}\n'.format(loss_w)
            # bktrk_report += 'constraint_val_w: {}\n'.format(constraint_val_w)
            # bktrk_report += 'loss_s: {}\n'.format(loss_s.value)
            # bktrk_report += 'constraint_val_s: {}\n'.format(constraint_val_s.value)
            # bktrk_report += 'loss_before_s: {}\n'.format(loss_before_s.value)
            # bktrk_report += '_max_constraint: {}\n'.format(self._max_constraint_val)
            # print bktrk_report

            if self._debug_nan and np.isnan(constraint_val_s):
                import ipdb; ipdb.set_trace()
            if loss_s.value < loss_before_s.value and constraint_val_s.value <= self._max_constraint_val:
                break

            # Give everybody time to compare before going back and resetting values.
            barrier_bktrk.wait()
        
        param_report = ''
        param_report += '\n Param Report [{}], itr: {}, n_iter: {}\n'.format(rank, itr, n_iter)
        param_report += 'prev_param[:3]: {}\n'.format(prev_param[:3])
        param_report += 'cur_param[:3]: {}\n'.format(cur_param[:3])
        print param_report

        # if rank == 0:
        logger.log("itr: %d    backtrack iters: %d" % (itr, n_iter))
            # logger.log("computing loss after")
            # logger.log("optimization finished")

        # nan_report = ''
        # nan_report += '\n NaN Report: [{}] END optimize_par \n'.format(rank)
        # param_check = self._target.get_param_values(trainable=True)
        # nan_report += 'param_check: {} / {}\n'.format(np.sum(np.isnan(param_check)), np.size(param_check))
        # print nan_report
