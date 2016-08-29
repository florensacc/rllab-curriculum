import joblib
import sys
import os

log_dirs = ["data/s3/trpo-momentum/20160610_165405_halfcheetah_trpo_m_none_bs_20k_T_500_s_1"]

for log_dir in log_dirs:
    for file_name in os.listdir(log_dir):
        if "cg_analysis" in file_name:
            file_name_full = os.path.join(log_dir,file_name)
            results = joblib.load(file_name_full)

            sys.exit(0)
