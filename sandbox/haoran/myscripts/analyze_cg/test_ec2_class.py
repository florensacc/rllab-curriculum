import os
from rllab import config

class TestClass(object):
    def __init__(self):
        pass
    def test(self,folder):
        file_name = "params.json"
        local_dir = "temp_data"
        os.system("mkdir %s"%(local_dir))
        remote_dir = os.path.join(config.AWS_S3_PATH, folder)
        remote_file = os.path.join(remote_dir,file_name)
        local_file = os.path.join(local_dir,file_name)
        command = """
            aws s3 cp {remote_file} {local_dir}/.
        """.format(remote_file=remote_file,local_dir=local_dir)
        os.system(command)
        with open(local_file,"r") as f:
            for line in f.readlines():
                print(line)
