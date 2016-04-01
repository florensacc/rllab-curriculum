import os.path as osp
import os

DOCKER_IMAGE = "quay.io/openai/rocky-sandbox"

KUBE_PREFIX = "rocky_"

DOCKER_LOG_DIR = "/tmp/expt"

AWS_S3_PATH = "s3://rocky-rllab-data/experiments"

AWS_IMAGE_ID = "ami-19343d73"

AWS_INSTANCE_TYPE = "m4.2xlarge"

AWS_KEY_NAME = "research_virginia"

AWS_SPOT = True

AWS_SPOT_PRICE = '1.0'

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY", None)

AWS_ACCESS_SECRET = os.environ.get("AWS_ACCESS_SECRET", None)

AWS_IAM_INSTANCE_PROFILE_NAME = "rllab"

AWS_SECURITY_GROUPS = ["rllab"]

AWS_REGION_NAME = "us-east-1"

CODE_SYNC_IGNORES = ["*.git/*", "*data/*"]
