# here we test accessing files stored on s3 from a ec2 instance
from rllab.misc.instrument import run_experiment_lite,stub
from myscripts.analyze_cg.test_ec2_class import TestClass
import os


stub(globals())
exp_prefix="test-ec2"
mode="ec2"
test_obj = TestClass()

folder = "trpo-momentum/20160610_165405_halfcheetah_trpo_m_none_bs_20k_T_500_s_1"

import datetime
import dateutil.tz
now = datetime.datetime.now(dateutil.tz.tzlocal())
timestamp = now.strftime('%Y%m%d_%H%M%S')
exp_name = "%s_test"%(timestamp)

run_experiment_lite(
    stub_method_call=test_obj.test(folder),
    exp_prefix=exp_prefix,
    exp_name=exp_name,
    script="scripts/run_experiment_lite.py",
    mode=mode,
    terminate_machine=False,
)
