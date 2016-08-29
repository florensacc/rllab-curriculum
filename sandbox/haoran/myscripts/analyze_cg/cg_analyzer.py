"""
analyze how different numbers of CG steps affect training
this class simply wraps all methods
"""

import theano
import theano.tensor as T
import numpy as np
import sys
sys.path.append('.')
from myscripts.myutilities import load_problem
import argparse
from rllab.sampler import parallel_sampler
from rllab.misc import ext
import time
from rllab.misc import krylov
import rllab.misc.logger as logger
import os
from rllab import config
import joblib

class CGAnalyzer(object):
    def __init__(self,
            log_dir,iteration,
            n_parallel=4,
            local_root="data/s3",
            train_batch_size=None,
            test_batch_size=80000,
            real_test_batch_size=80000,
            cg_iters = [np.inf,10,100,500]
        ):

        self.log_dir = log_dir
        self.iteration = iteration
        self.local_root = local_root
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.real_test_batch_size = real_test_batch_size
        self.cg_iters = cg_iters
        self.n_parallel=n_parallel

    def run(self):
        logger.log("Analyzing %s, iteration = %d"%(self.log_dir,self.iteration))

        # download the iteration snapshot from s3
        logger.log("Downloading snapshots and other files...")
        remote_dir = os.path.join(config.AWS_S3_PATH, self.log_dir)
        local_dir = os.path.join(self.local_root,self.log_dir)
        if not os.path.isdir(local_dir):
            os.system("mkdir -p %s"%(local_dir))

        file_names = [
            "itr_%d.pkl"%(self.iteration),
            "params.json",
            "progress.csv",
        ]
        for file_name in file_names:
            remote_file = os.path.join(remote_dir,file_name)
            command = """
                aws s3 cp {remote_file} {local_dir}/.
            """.format(remote_file=remote_file, local_dir=local_dir)
            os.system(command)

        # load problem ----------------------------------------------
        problem = load_problem(local_dir,self.iteration)
        algo = problem["algo"]
        algo.init_opt()
        if self.train_batch_size is None:
            self.train_batch_size = algo.batch_size

        # TRAIN PHASE ----------------------------------------------------
        # generate data
        self.pprint("Generating train data with size %d..."%(self.train_batch_size))
        train_inputs, train_data = self.collect_samples(algo, self.train_batch_size)

        # gradient
        train_g = algo.optimizer._opt_fun["f_grad"](*train_inputs)

        analysis = dict()
        analysis["train_g"] = train_g
        solns = dict()
        # exact natural gradient
        if np.inf in self.cg_iters:
            # compute the hessian and its condition number
            self.pprint("Computing the hessian matrix...")
            t = time.time()
            H = algo.optimizer._opt_fun["f_H"](*train_inputs)

            # debug code ...
            #n = len(train_g)
            #H = np.zeros((n,n))

            H_time = time.time() - t
            self.pprint("Total time %.3f seconds computing the Hessian"%(H_time))

            H_cond = np.linalg.cond(H)
            H_reg = H + algo.optimizer._reg_coeff * np.eye(H.shape[0])
            H_reg_cond = np.linalg.cond(H_reg)

            analysis["train_H"] = H
            analysis["train_H_time"] = H_time
            analysis["train_H_cond"] = H_cond
            analysis["train_H_reg_cond"] = H_reg_cond

            H_reg_invg = np.linalg.solve(H_reg,train_g)
            solns["H_reg_invg"] = dict()
            solns["H_reg_invg"]["direction"] = H_reg_invg


        # compute CG solutions for the given iterations
        def Hx(algo,x,inputs):
            xs = tuple(algo.policy.flat_to_params(x, trainable=True))
            plain = algo.optimizer._opt_fun["f_Hx_plain"](*(inputs + xs)) + algo.optimizer._reg_coeff * x
            return plain

        def train_Hx(x):
            return Hx(algo,x,train_inputs)

        def cg_callback(iteration,x):
            if iteration in self.cg_iters:
                soln_name = "cg_%d_soln"%(iteration)
                if soln_name not in solns:
                    solns[soln_name] = dict()
                    solns[soln_name]["direction"] = np.copy(x)
                    self.pprint("-- Finished iteration %d"%(iteration))

        cg_iters_new = self.cg_iters[:]
        if np.inf in cg_iters_new:
            cg_iters_new.remove(np.inf)
        if 0 not in cg_iters_new:
            cg_iters_new.append(0) # cgi=0 is just PG; useful for diagnosis
        max_cg_iter = np.amax(cg_iters_new)

        self.pprint("Computing CG solutions...")
        x_final, res_norms, soln_norms = krylov.cg(
            train_Hx,
            train_g,
            cg_iters=max_cg_iter+1,
            verbose=False,
            callback=cg_callback
        )

        # compare residual norms
        # should expect more CG, smaller norm
        for name,soln in solns.iteritems():
            soln["train_residual_norm"] = np.linalg.norm(train_g - train_Hx(soln["direction"]))

        # linear surrogate loss, quadratic constraint
        def compute_step_size(algo,inputs,direction):
            delta = algo.optimizer._max_constraint_val
            step_size = np.sqrt(2.0 * delta / (direction.dot(Hx(algo,direction,inputs))+1e-8))
            return step_size

        for name,soln in solns.iteritems():
            step_size = compute_step_size(algo,train_inputs,soln["direction"])
            soln["train_init_step"] = -step_size * soln["direction"]

        # surrogate loss, back tracking
        for name,soln in solns.iteritems():
            step, loss, constraint_val, n_iter = self.backtrack(algo,soln["train_init_step"],train_inputs)
            soln["train_surr_loss"] = loss
            soln["train_step"] = step
            soln["train_constraint"] = constraint_val
            soln["train_niter"] = n_iter

        # TEST PHASE --------------------------------------------------------------------
        # generate test data
        self.pprint("Generating test data with batch size %d..."%(self.test_batch_size))
        test_inputs,test_data = self.collect_samples(algo,batch_size=self.test_batch_size)

        # compute gradient
        test_g = algo.optimizer._opt_fun["f_grad"](*test_inputs)
        analysis["test_g"] = test_g

        if np.inf in self.cg_iters:
            test_H = algo.optimizer._opt_fun["f_H"](*test_inputs)
            analysis["test_H"] = test_H
        # linear cost, quadratic constraint
        def test_Hx(x):
            return Hx(algo,x,test_inputs)
        for name,soln in solns.iteritems():
            step_size = compute_step_size(algo,test_inputs,soln["direction"])
            soln["test_init_step"] = -step_size * soln["direction"]
            soln["test_residual_norm"] = np.linalg.norm(test_g - test_Hx(soln["direction"]))

        # surrogate loss, kl constraint
        for name,soln in solns.iteritems():
            step, loss, constraint, n_iter = self.backtrack(algo,soln["test_init_step"],test_inputs)
            soln["test_surr_loss"] = loss
            soln["test_step"] = step
            soln["test_constraint"] = constraint
            soln["test_niter"] = n_iter

        # REAL TEST PHASE --------------------------------------------------------------
        self.pprint("Generating real test data with batch size %d"%(self.real_test_batch_size))
        progress = problem["progress"]
        cur_average_discounted_return = progress["AverageDiscountedReturn"][self.iteration]
        cur_average_return = progress["AverageReturn"][self.iteration]
        for name,soln in solns.iteritems():
            self.pprint("-- Working on %s "%(name))
            average_return, average_discounted_return = self.compute_true_score(algo,soln["train_step"],batch_size=self.real_test_batch_size)
            soln["average_return"] = average_return
            soln["average_return_change"] = average_return - cur_average_return
            soln["average_discounted_return"] = average_discounted_return
            soln["average_discounted_return_change"] = average_discounted_return - cur_average_discounted_return

        # upload results to s3 -------------------------------------------------------
        results = dict(solns=solns,analysis=analysis)

        result_file = "itr_%d_cg_analysis.pkl"%(self.iteration)
        local_result_file = os.path.join(local_dir,result_file)
        joblib.dump(results, local_result_file, compress=3)
        logger.log("Results saved to %s"%(local_result_file))

        remote_result_file = os.path.join(remote_dir,result_file)
        os.system("""
            aws s3 cp {local_result_file} {remote_result_file}
        """.format(local_result_file=local_result_file, remote_result_file=remote_result_file))


    # HELPERS ------------------------------------------------------------------------------
    def collect_samples(self,algo,batch_size=None):
        algo.plot=False
        itr=0
        if batch_size is not None:
            algo.batch_size = batch_size

        parallel_sampler.initialize(self.n_parallel)
        algo.start_worker()
        paths = algo.obtain_samples(itr)
        samples_data = algo.process_samples(itr, paths)
        algo.shutdown_worker()

        all_input_values = tuple(ext.extract(
            samples_data,
            "observations", "actions", "advantages"
        ))
        agent_infos = samples_data["agent_infos"]
        state_info_list = [agent_infos[k] for k in algo.policy.state_info_keys]
        dist_info_list = [agent_infos[k] for k in algo.policy.distribution.dist_info_keys]
        all_input_values += tuple(state_info_list) + tuple(dist_info_list)

        return all_input_values,samples_data

    def backtrack(self,algo, proposed_step,all_input_values):
        optimizer = algo.optimizer
        optimizer._backtrack_ratio = 0.95
        optimizer._max_backtracks = 200

        loss_before = optimizer._opt_fun["f_loss"](*(all_input_values))
        prev_param = optimizer._target.get_param_values(trainable=True)
        for n_iter, ratio in enumerate(optimizer._backtrack_ratio ** np.arange(optimizer._max_backtracks)):
            cur_step = ratio * proposed_step
            cur_param = prev_param + cur_step
            optimizer._target.set_param_values(cur_param, trainable=True)
            loss, constraint_val = optimizer._opt_fun["f_loss_constraint"](*(all_input_values))
            if optimizer._debug_nan and np.isnan(constraint_val):
                import ipdb; ipdb.set_trace()
            if loss < loss_before and constraint_val <= optimizer._max_constraint_val:
                break
        optimizer._target.set_param_values(prev_param, trainable=True)
        return cur_step, loss, constraint_val, n_iter

    def compute_true_score(self,algo,step,batch_size=None):
        prev_param = algo.policy.get_param_values(trainable=True)
        cur_param = prev_param + step
        algo.policy.set_param_values(cur_param,trainable=True)
        _,samples = self.collect_samples(algo,batch_size)
        average_return = np.mean([sum(path["rewards"]) for path in samples["paths"]]) # average discounted return
        average_discounted_return = np.mean([path["returns"][0] for path in samples["paths"]])
        algo.policy.set_param_values(prev_param,trainable=True)
        return average_return, average_discounted_return

    def pprint(self,string):
        print '\033[93m' + string + '\033[0m'
