from rllab import config
import subprocess
import os.path as osp

def _rllab_root():
    import rllab
    return osp.abspath(osp.dirname(osp.dirname(rllab.__file__)))

def docker_build():
    root = _rllab_root()
    tag = config.DOCKER_IMAGE
    dockerfile_fullpath = osp.join(root, config.DOCKERFILE_PATH)
    subprocess.check_call(["docker", "build", "-t=%s"%tag, "-f", dockerfile_fullpath, root])
    
def docker_push():
    tag = config.DOCKER_IMAGE
    subprocess.check_call(["docker", "push", tag])

def docker_run():
    tag = config.DOCKER_IMAGE
    subprocess.check_call(["docker", "run", "-it", tag, "/bin/bash"])