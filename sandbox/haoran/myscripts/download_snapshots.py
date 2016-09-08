from rllab import config
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, default="trpo-momentum/20160610_165405_halfcheetah_trpo_m_none_bs_20k_T_500_s_1", nargs='?')
    parser.add_argument('--prompt',type=int,default=1)
    args = parser.parse_args()

    remote_dir = os.path.join(config.AWS_S3_PATH, args.folder)
    local_dir = os.path.join(config.LOG_DIR,"s3", args.folder)

    # by default ask for confirmation
    if args.prompt == 1:
        answer = input("Copying from \n{remote_dir} \nto \n{local_dir}\n (y/n)?".format(remote_dir=remote_dir,local_dir=local_dir))
        while answer not in ['y','Y','n','N']:
            print("Please input y(Y) or n(N)")
            answer = input("Are you sure? (y/n)")
    else:
        answer = 'y'

    if answer in ['y','Y']:
        os.system("""
            aws s3 cp {remote_dir} {local_dir} --recursive
        """.format(local_dir=local_dir, remote_dir=remote_dir))
        print("Download complete.")
    else:
        print("Abort download.")

    

