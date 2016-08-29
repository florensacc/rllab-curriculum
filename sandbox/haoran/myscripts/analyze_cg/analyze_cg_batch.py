from myscripts import analyze_cg
import joblib
import os
import numpy as np
import itertools
import sys

log_dirs = ["data/s3/trpo-momentum/alex_20160615_125712_halfcheetah_cgi100_bs20k_mnone_s1"]
iter_step = 10
iterations = np.arange(1,501,iter_step)

for log_dir, iteration in itertools.product(log_dirs, iterations):
    print "Analyzing %s, iteration = %d"%(log_dir,iteration) 
    results = analyze_cg.analyze(
        log_dir,
        iteration,
        train_batch_size=None,
        test_batch_size=80000,
        real_test_batch_size=80000,
        cg_iters=[10,100,500,1000],
    )
    result_file_name = os.path.join(log_dir,"itr_%d_cg_analysis.pkl"%(iteration))
    joblib.dump(results, result_file_name, compress=3)
    print "Results saved to %s"%(result_file_name)

