# analyze the CG iterations for walker
# I am interested in getting the train and test hessians for further analysis
from rllab.misc.instrument import run_experiment_lite,stub
from myscripts.analyze_cg.cg_analyzer import CGAnalyzer
import os,sys
import rllab.misc.logger as logger
import numpy as np


stub(globals())

exp_prefix="trpo-cg"
mode="local"
local_root = "data/s3"

log_dirs = [
    "trpo-momentum/20160610_165514_walker_trpo_m_none_bs_20k_T_500_s_1",
]
iter_step = 10
iterations = np.arange(0,500,iter_step)

train_batch_size=None
test_batch_size=80000
real_test_batch_size=80000
cg_iters = [10,100,500,1000,np.inf]

if mode=="local":
    n_parallel = 4
elif mode == "ec2":
    n_parallel = 1
else:
    raise NotImplementedError

for log_dir in log_dirs:
    # use batch_tasks may further equally distribute workload
    batch_tasks = [] 
    for iteration in iterations:
        analyzer=CGAnalyzer(
            log_dir,
            iteration,
            n_parallel=n_parallel,
            local_root=local_root,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            real_test_batch_size=real_test_batch_size,
            cg_iters=cg_iters,
        )
        task = dict(stub_method_call=analyzer.run())
        batch_tasks.append(task)

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
os.system("chmod 444 %s"%(__file__))

