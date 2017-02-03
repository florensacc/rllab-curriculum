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


        self.local_log_dir = osp.join(
            config.LOG_DIR,
            "local",
            self.exp_prefix.replace("_", "-"),
            self.exp_name,
        )
        if not osp.isdir(self.local_log_dir):
            os.system("mkdir -p %s"%(self.local_log_dir))
        self.remote_log_dir = osp.join(
            config.AWS_S3_PATH,
            self.exp_prefix.replace("_","-"),
            self.exp_name,
        )

    def grab_file(self, file_name):
        local_file = osp.join(self.local_log_dir, file_name)
        if osp.isfile(local_file):
            logger.log("Found file %s"%(local_file))
        else:
            logger.log("Downloading snapshot file...")
            remote_file = osp.join(self.remote_log_dir, file_name)
            try:
                os.system("""
                    aws s3 cp {remote} {local}
                """.format(
                    remote=remote_file,
                    local=local_file,
                ))
            except:
                print("Unable to download file %s"%(remote_file))
            logger.log("Download complete")

    def reload_snapshot(self):
        self.grab_file(self.snapshot_file)
        local_snapshot_file = osp.join(self.local_log_dir, self.snapshot_file)
        self.sess = tf.get_default_session() or tf_utils.create_session()
        self.snapshot = joblib.load(local_snapshot_file)
        try:
            self.algo = self.snapshot["algo"]
        except:
            print('Unable to load algo')

        print(self.configure_script)
        exec(self.configure_script)

    def reload_variant(self):
        self.grab_file("variant.json")
        local_variant_file = osp.join(self.local_log_dir, "variant.json")
        with open(local_variant_file) as vf:
            variant = json.load(vf)
        return variant

    def retrain(self):
        self.reload_snapshot()
        self.algo.train()
