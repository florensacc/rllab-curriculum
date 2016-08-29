# here we test accessing files stored on s3 from a ec2 instance
from rllab.misc.instrument import run_experiment_lite,stub
from myscripts.analyze_cg.cg_analyzer import CGAnalyzer
import os,sys
import rllab.misc.logger as logger
import numpy as np


stub(globals())

exp_prefix="trpo-cg"
mode="ec2"
local_root = "data/s3"

log_dirs = [
    "trpo-cg/alex_20160616_224333_halfcheetah_cgi0_bs20k_s1",
    "trpo-momentum/20160610_165405_halfcheetah_trpo_m_none_bs_20k_T_500_s_1",
    "trpo-momentum/alex_20160615_125712_halfcheetah_cgi100_bs20k_mnone_s1"

    "trpo-cg/alex_20160616_224305_walker_cgi0_bs20k_s1",
    "trpo-momentum/20160610_165514_walker_trpo_m_none_bs_20k_T_500_s_1",
    "trpo-momentum/alex_20160615_125635_walker_cgi100_bs20k_mnone_s1",

    "trpo-cg/alex_20160616_204439_ant_cgi0_bs20k_s1",
    "trpo-momentum/alex_20160615_144215_ant_cgi10_bs20k_mnone_s1",
    "trpo-momentum/alex_20160615_125752_ant_cgi100_bs20k_mnone_s1",

    "trpo-cg/alex_20160616_204043_hopper_cgi0_bs20k_s1",
    "trpo-momentum/alex_20160611_093848_hopper_m_none_bs_20k_s_1",
    "trpo-momentum/alex_20160615_125554_hopper_cgi100_bs20k_mnone_s1",
]
iter_step = 10
iterations = np.arange(0,500,iter_step)

n_parallel=1

train_batch_size=None
test_batch_size=80000
real_test_batch_size=80000
cg_iters = [10,100,500,1000]


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
