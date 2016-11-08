import os
from rllab import config
from rllab.viskit.core import load_exps_data
import numpy as np

from sandbox.rocky.neural_learner.scripts.visual_nav.utils import update_exp_pkl
from sandbox.rocky.s3.resource_manager import resource_manager

if __name__ == "__main__":

    exp_prefix = "doom-maze-77"

    exp_folder = os.path.join(config.PROJECT_PATH, "data/s3/%s" % exp_prefix)

    os.system("python scripts/sync_s3.py %s" % exp_prefix)

    exps = load_exps_data([exp_folder], ignore_missing_keys=True)

    best_exp = None
    best_score = None

    for exp in exps:
        if 'AverageReturn' not in exp.progress:
            continue
        score = np.mean(exp.progress['AverageReturn'][-10:])
        if score > -1.5:
            print(exp.params['exp_name'], score)
        if best_score is None or score > best_score:# and score < -1:
            best_score = score
            best_exp = exp

    best_exp_name = best_exp.params['exp_name']

    print("Best score:", best_score)
    print("Best exp:", best_exp_name)

    local_path = update_exp_pkl(exp_prefix, best_exp_name)

    # register on s3

    resource_name = "saved_params/doom-maze-v3/%s.pkl" % best_exp_name

    resource_manager.register_file(resource_name, local_path)

    print("Best resource name:", resource_name)
