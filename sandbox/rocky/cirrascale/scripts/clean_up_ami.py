from rllab import config
import boto3

REGIONS = [
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-south-1",
    "ap-southeast-1",
    "ap-southeast-2",
    "eu-central-1",
    "eu-west-1",
    "sa-east-1",
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
]


def get_clients():
    regions = REGIONS
    clients = []
    for region in regions:
        client = boto3.client(
            "ec2",
            region_name=region,
            aws_access_key_id=config.AWS_ACCESS_KEY,
            aws_secret_access_key=config.AWS_ACCESS_SECRET,
        )
        client.region = region
        clients.append(client)
    return clients


CLEANUP_SNAPSHOTS = True#False

if __name__ == "__main__":
    clients = get_clients()
    snapshot_ids = []
    if not CLEANUP_SNAPSHOTS:
        for client in clients:
            images = client.describe_images(Owners=['self'])['Images']
            to_delete = []
            for img in images:
                name = img['Name']
                if name.startswith("rllab-public") or name.startswith("rllab-packer"):
                    to_delete.append(img)
                    print('Deleting {name} in {region}'.format(name=name, region=client.region))
                    snapshot_ids.append((client.region, img['BlockDeviceMappings'][0]['Ebs']['SnapshotId']))
                    print(snapshot_ids)
                    client.deregister_image(ImageId=img['ImageId'])  # , delete_snapshot=True)
    else:
        snapshot_ids = [('ap-northeast-1', 'snap-30263d0b'), ('ap-northeast-1', 'snap-4110e4cf'), ('ap-northeast-2',
                                                                                                   'snap-2b8fd4c2'),
                        ('ap-northeast-2', 'snap-c8c6c621'), ('ap-south-1', 'snap-92a3e17b'),
                        ('ap-south-1', 'snap-f228511b'), ('ap-southeast-1', 'snap-a8e8e42d'),
                        ('ap-southeast-1', 'snap-98dd711d'), ('ap-southeast-2', 'snap-5aa13aca'),
                        ('ap-southeast-2', 'snap-84b8668a'), ('eu-central-1', 'snap-bbdfb4c2'),
                        ('eu-central-1', 'snap-030ebde8'), ('eu-west-1', 'snap-6a8d0d9d'),
                        ('eu-west-1', 'snap-efbe5789'), ('sa-east-1', 'snap-ab9f0fbb'), ('sa-east-1', 'snap-01492a11'),
                        ('us-east-1', 'snap-32230dba'), ('us-east-1', 'snap-8f913912'), ('us-east-2', 'snap-531bbda2'),
                        ('us-east-2', 'snap-7c38e88d'), ('us-west-1', 'snap-724adcf3'), ('us-west-1', 'snap-01d27c81'),
                        ('us-west-2', 'snap-47d67d63'), ('us-west-2', 'snap-e4816cb2')]
        for client in clients:
            for region, snapshot_id in snapshot_ids:
                if client.region == region:
                    print("Deleting {snapshot_id}".format(snapshot_id=snapshot_id))
                    client.delete_snapshot(SnapshotId=snapshot_id)
