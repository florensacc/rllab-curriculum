import argparse
import os
from rllab import config
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, default=None, nargs='?')
    args = parser.parse_args()
    remote_dir = "{fw}:~/rllab-master/rllab/data/s3".format(
        fw="haoran@finallyworking.banatao.berkeley.edu")
    local_dir = os.path.join(config.LOG_DIR,"s3")
    if args.folder:
        remote_dir = os.path.join(remote_dir,args.folder)
        local_dir = os.path.join(local_dir,args.folder)

    file_names = ["params.json","progress.csv"]
    file_names += ["itr_%d.pkl"%(i) for i in np.arange(1,501,10)]
    for file_name in file_names:
        command = """
            scp -r {local_dir}/{file_name} {remote_dir} 
        """.format(local_dir=local_dir,remote_dir=remote_dir,file_name=file_name)
        signal = os.system(command)
        if signal != 0:
            break
