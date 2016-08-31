from myscripts.myutilities import load_problem
from rllab.misc.special import ask_permission
from rllab.sampler import parallel_sampler
from rllab.misc import ext
from rllab.misc import krylov
import rllab.misc.logger as logger
from rllab import config

import argparse
import time
import os
import joblib
import theano
import theano.tensor as T
import numpy as np
import sys
sys.path.append('.')

class RegAnalyzer(object):
    """
    A set of methods for analyzing the best regularizer (for each iteration).
    Collect many samples; compute the almost exact FIM and PG; estimate various heuristics of the best regularizer by bootstrapping sample PG. Sample PG may be used to construct the sample NPG or CG solutions.
    Store all samples and FIM on s3, to allows for further analysis.

    Heuristics:
    1. The expected (squared) distance to the true natural gradient.
    2. The expected (squared) distance to the scaled true natural gradient.
    3. A lower confidence bound on the surrogate loss (evaluated on the huge batch).
    4. A lower confidence bound on the true loss (by further sampling)
    """
    def __init__(self,
            log_dir,iteration,
            local_root="data/s3",
            n_parallel=4,

            cur_batch_size=100000,
            F_subsample_size=10000,
            true_npg_reg=1e-15,

            bootstrap_subsample_size=10000,
            bootstrap_set_count=100,
            cg_iters_list=[np.inf],
            reg_list=[1e-5],

            next_batch_size=100000,
        ):
        """
        :param n_parallel: number of parallel workers for sampling paths

        :param cur_batch_size: number of samples we obtain that should give a very accurate estimate of PG
        :param F_subsample_size: number of samples we draw from cur_batch to compute FIM. It should be large enough to make the sample FIM accurate, but not too large to slow down computation.
        :param true_npg_reg: a (tiny) regularization term we add to the FIM when computing the "true" natural gradient

        :param bootstrap_subsample_size: size of each subsample that we draw (without replacement) from cur_batch.
        :param bootstrap_set_count: number of bootstrapped subsample sets.
        :param cg_iters:
        """
        self.log_dir = log_dir
        self.iteration = iteration
        self.local_root = local_root
        self.n_parallel = n_parallel
        self.cur_batch_size = cur_batch_size
        self.F_subsample_size = F_subsample_size
        self.true_npg_reg = true_npg_reg
        self.bootstrap_subsample_size = bootstrap_subsample_size
        self.bootstrap_set_count = bootstrap_set_count
        self.cg_iters_list = cg_iters_list
        self.reg_list = reg_list
        self.next_batch_size = next_batch_size
        self.data = dict()

        self.local_dir = os.path.join(self.local_root,self.log_dir)
        self.remote_dir = os.path.join(config.AWS_S3_PATH, self.log_dir)
        data_file = "itr_%d_data.pkl"%(self.iteration)
        self.local_data_file = os.path.join(self.local_dir, data_file)
        self.remote_data_file = os.path.join(self.remote_dir, data_file)

        self.sub_g_name = "sub_%d_g_list"%(self.bootstrap_subsample_size)

    def load_snapshot(self):
        """
        Download the iteration snapshot from s3 to local. And then load it.
        """
        logger.log("Downloading snapshots and other files...")
        if not os.path.isdir(self.local_dir):
            os.system("mkdir -p %s"%(self.local_dir))

        file_names = [
            "itr_%d.pkl"%(self.iteration),
            "params.json",
            "progress.csv",
        ]
        for file_name in file_names:
            remote_file = os.path.join(self.remote_dir,file_name)
            command = """
                aws s3 cp {remote_file} {local_dir}/.
            """.format(remote_file=remote_file, local_dir=self.local_dir)
            os.system(command)

        self.snapshot = load_problem(self.local_dir,self.iteration)
        self.pprint("Finished loading the snapshot.")


    def prepare_data(self):
        logger.log("Preparing data for %s, iteration = %d"%(self.log_dir,self.iteration))

        # generate many paths
        algo = self.snapshot["algo"]
        algo.init_opt()
        self.pprint("Generating data with size %d by the current policy..."%(self.cur_batch_size))
        cur_inputs, cur_data = self.collect_samples(algo, self.cur_batch_size)

        # true gradient
        f_grad = algo.optimizer._opt_fun["f_grad"]
        g = f_grad(*cur_inputs)

        # true FIM
        self.pprint("Computing the FIM")
        t = time.time()
        f_H = algo.optimizer._opt_fun["f_H"]
        F_inputs = self.subsample_inputs(cur_inputs,self.F_subsample_size)
        F = f_H(*F_inputs)
        self.pprint("Finished computing the FIM within %.1f seconds"%(time.time() - t))

        # eigen-decomposition of F
        w,v  = np.linalg.eig(F)
        w = np.real(w)
        v = np.real(v)

        # compute a more accurate estimate of g_nat with negligible regularization
        g_nat = v.dot(np.diag(1./(w + self.true_npg_reg))).dot(v.T).dot(g)

        # store the data to local_dir
        self.data["g"] = g
        self.data["F"] = F
        self.data["F_e_values"] = w
        self.data["F_e_vectors"] = v
        self.data["g_nat"] = g_nat
        self.data["cur_inputs"] = cur_inputs
        self.data["snapshot"] = self.snapshot


    def store_and_upload_data(self,force_merge=False):
        joblib.dump(self.data,self.local_data_file,compress=3)

        # download and merge with any already computed datum from s3 (if it is not in self.data)
        local_data_file_tmp = os.path.join(self.local_dir, "itr_%d_data_tmp.pkl"%(self.iteration))
        msg = os.system("aws s3 cp %s %s"%(self.remote_data_file, local_data_file_tmp))
        if msg == 0: # downloaded, which also means the remote data file exists
            if force_merge:
                merge = True
            else:
                merge = ask_permission("The remote data file exists. Do you want to merge with it (y/Y)?")
            if merge:
                data_tmp = joblib.load(local_data_file_tmp)
                for k,v in data_tmp.items():
                    if k not in self.data:
                        self.data[k] = v

            os.system("rm %s"%(local_data_file_tmp))

        # upload data to s3
        os.system("aws s3 cp %s %s"%(self.local_data_file, self.remote_data_file))

    def run_first(self):
        self.load_snapshot()
        self.prepare_data()
        self.store_and_upload_data(force_merge=True)

    # ----------------------------------------------------------------------
    def reload_data_file(self):
        if not os.path.exists(self.local_data_file):
            os.system("aws s3 cp %s %s"%(self.remote_data_file, self.local_data_file))
        self.data = joblib.load(self.local_data_file)


    def bootstrap_gradients(self):
        algo = self.data["snapshot"]["algo"]
        algo.init_opt()
        f_grad = algo.optimizer._opt_fun["f_grad"]
        # bootstrap a bunch of subsampled gradients
        sub_g_list = self.subsample_grad(
            f_grad,
            self.data["cur_inputs"],
            self.bootstrap_subsample_size,
            self.bootstrap_set_count
        )
        self.data[self.sub_g_name] = sub_g_list

    def compute_descent_directions(self):
        # exclude np.inf
        cg_iters_list_finite = self.cg_iters_list[:]
        if np.inf in cg_iters_list_finite:
            cg_iters_list_finite.remove(np.inf)

        # prepare empty lists
        for reg in self.reg_list:
            for itr in cg_iters_list_finite:
                k = "sub_%d_reg_%.2e_cg_%d_list"%(self.bootstrap_subsample_size,reg,itr)
                self.data[k] = []
            if np.inf in self.cg_iters_list:
                k = "sub_%d_reg_%.2e_g_nat_list"%(self.bootstrap_subsample_size,reg)
                self.data[k] = []

        # compute the solutions
        for sub_g in self.data[self.sub_g_name]:
            for reg in self.reg_list:
                # cg solutions
                F = self.data["F"]
                F_reg = F + reg * np.eye(F.shape[0])
                solns = self.compute_cg_solns(
                    H=F_reg,
                    g=sub_g,
                    cg_iters_list=cg_iters_list_finite,
                )

                for itr, soln in solns.items():
                    k = "sub_%d_reg_%.2e_cg_%d_list"%(self.bootstrap_subsample_size,reg,itr)
                    self.data[k].append(soln)

                # natural gradients
                if np.inf in self.cg_iters_list:
                    w = self.data["F_e_values"]
                    v = self.data["F_e_vectors"]
                    sub_g_nat = v.dot(np.diag(1./(w+reg))).dot(v.T).dot(sub_g)
                    k = "sub_%d_reg_%.2e_g_nat_list"%(self.bootstrap_subsample_size,reg)
                    self.data[k].append(sub_g_nat)

    def compute_cg_solns(self, H, g, cg_iters_list):
        cg_iters_max = np.amax(cg_iters_list)
        def Hx(x):
            return H.dot(x)
        solns = dict()
        def callback(itr,x):
            if itr in cg_iters_list:
                solns[itr] = np.copy(x)
                self.pprint("-- Getting CG solution at iteration %d"%(itr))

        x, residual_norms, soln_norms = krylov.cg(Hx, g, cg_iters=cg_iters_max, callback=callback)
        return solns

    def analyze_descent_directions(self, stats=None):
        algo = self.snapshot["algo"]
        algo.init_opt()
        F = self.data["F"]
        g = self.data["g"]
        g_nat = self.data["g_nat"]

        def rescale_direction(d):
            return d * 1./np.sqrt(d.dot(F.dot(d)))
        g_nat_scaled = rescale_direction(g_nat)
        self.data["g_nat_scaled"] = g_nat_scaled

        if stats is None:
            stats = ["dist","scaled_dist"]
        if "surr_loss" in stats:
            stats.append("step_dist")
            g_nat_step, surr_loss, constraint_val, _, _=                     self.line_search(algo,-g_nat_scaled,self.data["cur_inputs"])
            self.data["g_nat_step"] = g_nat_step
        # compute the relevant statistics
        for reg in self.reg_list:
            for cg_iters in self.cg_iters_list:
                if cg_iters != np.inf:
                    k = "sub_%d_reg_%.2e_cg_%d"%(self.bootstrap_subsample_size,reg,cg_iters)
                else:
                    k = "sub_%d_reg_%.2e_g_nat"%(self.bootstrap_subsample_size,reg)
                for stat in stats:
                    self.data[k + "_" + stat] = []

                directions = self.data[k + "_list"]

                if "scaled_dist" in stats:
                    self.data[k + "_scaled"] = []
                    if "surr_loss" in stats:
                        self.data[k + "_step"] = []

                for d in directions:
                    # distance
                    dist = np.linalg.norm(d-g_nat)

                    if "scaled_dist" in stats:
                        # rescale, and then l2 distance
                        d_scaled = rescale_direction(d)
                        self.data[k + "_scaled"].append(d_scaled)
                        scaled_dist = np.linalg.norm(d_scaled - g_nat_scaled)

                        # line search and then compute surrogate loss
                        if "surr_loss" in stats:
                            d_step, surr_loss, constraint_val, _, _=                     self.line_search(algo,-d_scaled,self.data["cur_inputs"])
                            self.data[k + "_step"].append(d_step)
                            step_dist = np.linalg.norm(d_step - g_nat_step)

                            # take the step informed by line search; then resample
                            if "avg_return" in stats:
                                avg_return, avg_discounted_return =  self.compute_true_score(algo,d_step,self.next_batch_size)

                    for stat in stats:
                        self.data[k + "_" + stat].append(locals()[stat])
                for stat in stats:
                    self.data[k + "_" + stat + "_mean"] = np.average(self.data[k + "_" + stat])
                    self.data[k + "_" + stat + "_std"] =                      np.std(self.data[k + "_" + stat])

    def run_second(self,stats=None):
        self.load_snapshot()
        self.reload_data_file()
        self.bootstrap_gradients()
        self.compute_descent_directions()
        self.analyze_descent_directions(stats)
        self.store_and_upload_data(force_merge=True)





    # HELPERS ------------------------------------------------------------------------------
    def collect_samples(self,algo,batch_size):
        algo.plot=False
        itr=0
        algo.batch_size = batch_size
        parallel_sampler.initialize(self.n_parallel)
        algo.start_worker()
        paths = algo.obtain_samples(itr,phase="train")
        samples_data = algo.process_samples(itr, paths, phase="train")
        algo.shutdown_worker()

        all_input_values = algo.prepare_optimizer_inputs(samples_data)

        return all_input_values,samples_data

    def line_search(self,algo, flat_descent_step,all_input_values):
        self.pprint("Starting line search")
        optimizer = algo.optimizer
        optimizer._foretrack_ratio = 1.05
        optimizer._backtrack_ratio = 0.95
        optimizer._max_backtracks = 100
        optimizer._max_foretracks = 100

        loss_before = optimizer._opt_fun["f_loss"](*(all_input_values))

        # foretracking until the objective is increased or the constraint is violated
        prev_param = optimizer._target.get_param_values(trainable=True)
        fore_iter = 0
        fore_ratio = 1
        fore_iters = np.arange(1, 1 + optimizer._max_foretracks)
        for fore_iter, fore_ratio in zip(fore_iters, optimizer._foretrack_ratio ** fore_iters):
            cur_step = fore_ratio * flat_descent_step
            cur_param = prev_param + cur_step
            optimizer._target.set_param_values(cur_param, trainable=True)
            loss, constraint_val = optimizer._opt_fun["f_loss_constraint"](*all_input_values)
            if optimizer._debug_nan and np.isnan(constraint_val):
                import ipdb; ipdb.set_trace()
            if not (loss < loss_before and constraint_val <= optimizer._max_constraint_val):
                fore_iter = fore_iter + 1
                break
        flat_descent_step = flat_descent_step * fore_ratio

        # backtracking to ensure decreasing the objective and obeying the constraint
        for back_iter, back_ratio in enumerate(optimizer._backtrack_ratio ** np.arange(optimizer._max_backtracks)):
            cur_step = back_ratio * flat_descent_step
            cur_param = prev_param + cur_step
            optimizer._target.set_param_values(cur_param, trainable=True)
            loss, constraint_val = optimizer._opt_fun["f_loss_constraint"](*(all_input_values))
            if optimizer._debug_nan and np.isnan(constraint_val):
                import ipdb; ipdb.set_trace()

            satisfy_loss = loss < optimizer
            satisfy_constraint = constraint_val <= optimizer._max_constraint_val
            if satisfy_loss and satisfy_constraint:
                break
        optimizer._target.set_param_values(prev_param, trainable=True)
        self.pprint("-- Line search finished")
        return cur_step, loss, constraint_val, fore_iter, back_iter


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

    def subsample_inputs(self,inputs,subsample_size):
        for x in inputs:
            assert isinstance(x,np.ndarray)
        total_batch_size = inputs[0].shape[0]
        indices = np.random.randint(low=0,high=total_batch_size,size=subsample_size)

        sub_inputs = []
        for x in inputs:
            if len(x.shape) == 1: # vector
                sub_x = np.array([x[i] for i in indices])
            elif len(x.shape) == 2: #matrix
                sub_x = np.vstack([x[i] for i in indices])
            else:
                raise NotImplementedError
            sub_inputs.append(sub_x)

        return tuple(sub_inputs)

    def subsample_grad(self,f_grad,inputs,bootstrap_subsample_size, bootstrap_set_count):
        from rllab.sampler.stateful_pool import ProgBarCounter
        sub_g_list = []
        pbar = ProgBarCounter(bootstrap_set_count)
        for i in range(bootstrap_set_count):
            sub_inputs = self.subsample_inputs(inputs,bootstrap_subsample_size)
            sub_g_list.append(f_grad(*sub_inputs))
            pbar.inc(1)
        pbar.stop()
        return sub_g_list
