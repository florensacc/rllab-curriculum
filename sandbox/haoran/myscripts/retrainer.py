import os
import os.path as osp
import joblib
from rllab import config
from rllab.misc import logger

class Retrainer(object):
    def __init__(self,
        exp_prefix,
        exp_name,
        snapshot_file,
        target_itr,
        n_parallel,
        configure_script,
    ):
        """
        :param s3_pkl_file: name of the pkl file, excluding dir name
        :param configure_script: any command that account for code changes in order to make the code run again
        """
        self.exp_prefix = exp_prefix
        self.exp_name = exp_name
        self.snapshot_file = snapshot_file
        self.target_itr = target_itr
        self.n_parallel = n_parallel
        self.configure_script = configure_script

    def reload_snapshot(self):
        local_log_dir = osp.join(
            config.LOG_DIR,
            "local",
            self.exp_prefix.replace("_", "-"),
            self.exp_name,
        )
        if not osp.isdir(local_log_dir):
            os.system("mkdir -p %s"%(local_log_dir))
        local_snapshot_file = osp.join(local_log_dir, self.snapshot_file)
        if osp.isfile(local_snapshot_file):
            logger.log("Found snapshot file %s"%(local_snapshot_file))
        else:
            logger.log("Downloading snapshot file...")
            remote_log_dir = osp.join(
                config.AWS_S3_PATH,
                self.exp_prefix.replace("_","-"),
                self.exp_name,
            )
            remote_snapshot_file = osp.join(remote_log_dir, self.snapshot_file)
            try:
                os.system("""
                    aws s3 cp {remote} {local}
                """.format(
                    remote=remote_snapshot_file,
                    local=local_snapshot_file,
                ))
            except:
                print("Unable to download snapshot file %s"%(remote_snapshot_file))
            logger.log("Download complete")

        self.snapshot = joblib.load(local_snapshot_file)
        self.algo = self.snapshot["algo"]
        print("Algo last itr: %d, target itr: %d"%(
            self.algo.current_itr,
            self.target_itr,
        ))
        self.algo.current_itr += 1
        self.algo.n_itr = self.target_itr
        self.algo.n_parallel = self.n_parallel

        exec(self.configure_script)

    def retrain(self):
        self.reload_snapshot()
        self.algo.train()
