from rllab import config
import os
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        extra = " ".join(sys.argv[1:])
    else:
        extra = ""
    os.system("""
        aws s3 sync {remote_dir} {local_dir} --exclude '*debug.log'  --exclude '*.pkl' --content-type "UTF-8" {extra}
    """.format(local_dir=os.path.join(config.LOG_DIR, "s3"), remote_dir=config.AWS_S3_PATH,
               extra=extra))
