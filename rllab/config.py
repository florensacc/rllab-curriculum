import os.path as osp
import os

PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))

LOG_DIR = PROJECT_PATH + "/data"

USE_TF = False

DOCKER_IMAGE = "DOCKER_IMAGE"

DOCKERFILE_PATH = "/path/to/Dockerfile"

KUBE_PREFIX = "rllab_"

DOCKER_LOG_DIR = "/tmp/expt"

POD_DIR = PROJECT_PATH + "/.pods"

AWS_S3_PATH = None

AWS_IMAGE_ID = None

AWS_INSTANCE_TYPE = "m4.xlarge"

AWS_KEY_NAME = "AWS_KEY_NAME"

AWS_SPOT = True

AWS_SPOT_PRICE = '1.0'

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY", None)

AWS_ACCESS_SECRET = os.environ.get("AWS_ACCESS_SECRET", None)

AWS_IAM_INSTANCE_PROFILE_NAME = "rllab"

AWS_SECURITY_GROUPS = ["rllab"]

AWS_SECURITY_GROUP_IDS = []

AWS_NETWORK_INTERFACES = []

AWS_EXTRA_CONFIGS = dict()

AWS_REGION_NAME = "us-east-1"

AWS_BUCKET_REGION_NAME = "us-east-1"  # added by me trying to fix the issue.

CODE_SYNC_IGNORES = ["*.git/*", "*data/*", "*.pod/*"]

DOCKER_CODE_DIR = "/root/code/rllab"

AWS_CODE_SYNC_S3_PATH = "s3://to/be/overriden/in/personal"

# whether to use fast code sync
FAST_CODE_SYNC = True

FAST_CODE_SYNC_IGNORES = [".git", "data", ".pods"]

KUBE_DEFAULT_RESOURCES = {
    "requests": {
        "cpu": 0.8,
    }
}

KUBE_DEFAULT_NODE_SELECTOR = {
    "aws/type": "m4.xlarge",
}

MUJOCO_KEY_PATH = osp.expanduser("~/.mujoco")
# MUJOCO_KEY_PATH = osp.join(osp.dirname(__file__), "../vendor/mujoco")

ENV = {}

EBS_OPTIMIZED = True

if osp.exists(osp.join(osp.dirname(__file__), "config_personal.py")):
    from .config_personal import *
else:
    print("Creating your personal config from template...")
    from shutil import copy
    copy(osp.join(PROJECT_PATH, "rllab/config_personal_template.py"), osp.join(PROJECT_PATH, "rllab/config_personal.py"))
    from .config_personal import *
    print("Personal config created, but you should probably edit it before further experiments " \
          "are run")
    if 'CIRCLECI' not in os.environ:
        print("Exiting.")
        import sys; sys.exit(0)

LABEL = ""

INSTANCE_TYPE_INFO = {  #this prices are orientative.
    "c4.large": dict(price=0.105, vCPU=2),
    "c4.xlarge": dict(price=0.209, vCPU=4),
    "c4.2xlarge": dict(price=0.419, vCPU=8),
    "c4.4xlarge": dict(price=1.338, vCPU=16),
    "c4.8xlarge": dict(price=1.60, vCPU=36),
    "m4.large": dict(price=0.1, vCPU=2),
    "m4.xlarge": dict(price=0.5, vCPU=4),
    "m4.2xlarge": dict(price=0.5, vCPU=8),
    "m4.4xlarge": dict(price=0.8, vCPU=16),
    "m4.10xlarge": dict(price=2.394, vCPU=40),
    "m4.16xlarge": dict(price=1.5, vCPU=64),
    "g2.2xlarge": dict(price=0.65, vCPU=8),
}
