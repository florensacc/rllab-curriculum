import os
import os.path as osp
import joblib
import tensorflow as tf
from rllab import config
from rllab.misc import logger
from sandbox.haoran.myscripts import tf_utils

class Retrainer(object):
    def __init__(self,
        exp_prefix,
        exp_name,
        snapshot_file,
        configure_script,
    ):
        """
        :param s3_pkl_file: name of the pkl file, excluding dir name
        :param configure_script: any command that account for code changes in order to make the code run again
        """
        self.exp_prefix = exp_prefix
        self.exp_name = exp_name
        self.snapshot_file = snapshot_file
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

        self.sess = tf.get_default_session() or tf_utils.create_session()
        self.snapshot = joblib.load(local_snapshot_file)
        try:
            self.algo = self.snapshot["algo"]
        except:
            print('Unable to load algo')

        print(self.configure_script)
        exec(self.configure_script)

    def retrain(self):
        self.reload_snapshot()
        self.algo.train()
