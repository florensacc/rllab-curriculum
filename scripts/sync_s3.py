from rllab import config
import boto3
import os
from urlparse import urlparse

def download_dir(client, resource, dist, local='/tmp', bucket='your_bucket'):
    paginator = client.get_paginator('list_objects')
    for result in paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=dist):
        if result.get('CommonPrefixes') is not None:
            for subdir in result.get('CommonPrefixes'):
                download_dir(client, resource, subdir.get('Prefix'), local, bucket=bucket)
        if result.get('Contents') is not None:
            for file in result.get('Contents'):
                if not os.path.exists(local + os.sep + file.get('Key')):
                    if not os.path.exists(os.path.dirname(local + os.sep + file.get('Key'))):
                         os.makedirs(os.path.dirname(local + os.sep + file.get('Key')))
                    print "Downloading %s" % file.get('Key')
                    resource.meta.client.download_file(bucket, file.get('Key'), local + os.sep + file.get('Key'))

if __name__ == "__main__":

    client = boto3.client(
        "s3",
        region_name=config.AWS_REGION_NAME,
        aws_access_key_id=config.AWS_ACCESS_KEY,
        aws_secret_access_key=config.AWS_ACCESS_SECRET,
    )

    resource = boto3.resource(
        "s3",
        region_name=config.AWS_REGION_NAME,
        aws_access_key_id=config.AWS_ACCESS_KEY,
        aws_secret_access_key=config.AWS_ACCESS_SECRET,
    )

    parsed = urlparse(config.AWS_S3_PATH)
    bucket = parsed.netloc
    dist = parsed.path[1:] + "/"
    download_dir(client, resource, dist=dist, local=os.path.join(config.LOG_DIR, "s3"), bucket=bucket)
