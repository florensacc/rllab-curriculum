


import os
import glob
import shutil
from rllab import config
from rllab.viskit.core import load_progress, load_params, flatten_dict

if __name__ == "__main__":

    root_path = os.path.join(config.PROJECT_PATH, "data/s3/async-ddpg-final-search")


    dirs = glob.glob(os.path.join(root_path, "*"))

    cnt = 0

    for dir in dirs:
        try:
            params_json_path = os.path.join(dir, "params.json")
            progress_csv_path = os.path.join(dir, "progress.csv")
            progress = load_progress(progress_csv_path)
            params = load_params(params_json_path)
            flat_params = flatten_dict(params)
            exp_name = flat_params["exp_name"]
            max_samples = flat_params["json_args.algo.max_samples"]
            actual_samples = int(progress["NSamples"][-1])
            if max_samples == actual_samples:
                cnt += 1
            else:
                shutil.move(dir, os.path.join(dir, "../../async-ddpg-final-search-incomplete", exp_name))
        except IOError:
            exp_name = dir.split("/")[-1]
            shutil.move(dir, os.path.join(dir, "../../async-ddpg-final-search-incomplete", exp_name))
    print(cnt)
