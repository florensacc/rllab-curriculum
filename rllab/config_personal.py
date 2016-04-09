import os.path as osp
import os


USE_GPU = False

#DOCKER_IMAGE = "quay.io/openai/rocky-sandbox-gpu"
DOCKER_IMAGE = "rein/rllab-exp-new"#neocxi/rllab:atari

KUBE_PREFIX = "rhouthooft_"

DOCKER_LOG_DIR = "/tmp/expt"

AWS_S3_PATH = "s3://openai-kubernetes-sci-rein/experiments"

AWS_IMAGE_ID = "ami-67c5d00d"

if USE_GPU:
    AWS_INSTANCE_TYPE = "g2.2xlarge"
else:
    AWS_INSTANCE_TYPE = "m4.2xlarge"

AWS_KEY_NAME = "research_virginia"

AWS_SPOT = True

if USE_GPU:
    AWS_SPOT_PRICE = '10.0'
else:
    AWS_SPOT_PRICE = '1.0'

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY", None)

AWS_ACCESS_SECRET = os.environ.get("AWS_ACCESS_SECRET", None)

AWS_IAM_INSTANCE_PROFILE_NAME = "rllab"

AWS_SECURITY_GROUPS = ["rllab"]

AWS_REGION_NAME = "us-west-2"

AWS_CODE_SYNC_S3_PATH = "s3://openai-kubernetes-sci-rein/code"

#CODE_SYNC_IGNORES = ["*.git/*", "*data/*"]
CODE_SYNC_IGNORES = ["*.git/*", "*data/*", "*src/*"]

LOCAL_CODE_DIR = "/home/rein/workspace_python/rllab"

LABEL = "rhouthooft"

DOCKER_CODE_DIR = "/root/code/rllab"
