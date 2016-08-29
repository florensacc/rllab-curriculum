# Simply compute the Hessian and sample a bunch of gradients

from rllab.misc.instrument import run_experiment_lite,stub
from myscripts.analyze_reg.reg_analyzer import RegAnalyzer
import rllab.misc.logger as logger
import os,sys
import numpy as np
import itertools


stub(globals())

exp_prefix="trpo-off-policy"
mode="ec2"
local_root = "data/s3"

#--------------------------------------------------------
# general params
cur_batch_size = 100000
F_subsample_size = 10000
true_npg_reg=1e-8
bootstrap_set_count=100

next_batch_size=10

# product params
log_dirs = [
    "trpo-momentum/20160610_165514_walker_trpo_m_none_bs_20k_T_500_s_1",
    "trpo-momentum/20160610_141900_hopper_trpo_momentum_none_bs_1k_T_500_s_1",
    "trpo-momentum/20160610_145336_swimmer_trpo_m_none_bs_20k_T_500_s_1",
]
itr_gap = 5
iterations = np.arange(0,300+itr_gap,itr_gap)
bootstrap_subsample_size_list = [1000]
cg_iters_list=[10]
reg_list=[0.1]

if mode=="local":
    n_parallel = 4
elif mode == "ec2":
    n_parallel = 1
else:
    raise NotImplementedError

# -------------------------------------------------------
for log_dir,iteration in itertools.product(log_dirs, iterations):
    # use batch_tasks may further equally distribute workload
    batch_tasks = []
    first_done = False
    for bootstrap_subsample_size in bootstrap_subsample_size_list:
        analyzer=RegAnalyzer(
            log_dir,iteration,
            local_root=local_root,
            n_parallel=n_parallel,

            cur_batch_size=cur_batch_size,
            F_subsample_size=F_subsample_size,
            true_npg_reg=true_npg_reg,

            bootstrap_subsample_size=bootstrap_subsample_size,
            bootstrap_set_count=bootstrap_set_count,
            cg_iters_list=cg_iters_list,
            reg_list=reg_list,

            next_batch_size=next_batch_size,
        )
        if not first_done:
            batch_tasks.append(dict(stub_method_call=analyzer.run_first()))
            first_done = True
        # batch_tasks.append(dict(stub_method_call=analyzer.run_second()))

    import datetime
    import dateutil.tz
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y%m%d_%H%M%S')
    exp_name = "alex_%s_analyze_cg"%(timestamp)

    run_experiment_lite(
        exp_prefix=exp_prefix,
        exp_name=exp_name,
        batch_tasks=batch_tasks,
        mode=mode,
        terminate_machine=True,
    )
    sys.exit(0)

if mode == "ec2":
    os.system("chmod 444 %s"%(__file__))
