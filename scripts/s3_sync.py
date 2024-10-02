import os
import boto3
from botocore.exceptions import ClientError
import logging
from tqdm import tqdm
import argparse

# Set up logging
logging.basicConfig(filename='logs/s3_sync.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize S3 client
s3 = boto3.client('s3')

# S3 bucket name
BUCKET_NAME = 'dna-xformers-yelab'

def upload_file(file_path, bucket, object_name=None):
    """Upload a file to an S3 bucket"""
    if object_name is None:
        object_name = file_path

    try:
        with tqdm(total=os.path.getsize(file_path), unit='B', unit_scale=True, desc=file_path) as pbar:
            s3.upload_file(
                file_path, 
                bucket, 
                object_name,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
            )
    except ClientError as e:
        logging.error(e)
        return False
    return True

def download_file(bucket, object_name, file_path):
    """Download a file from an S3 bucket"""
    try:
        with tqdm(unit='B', unit_scale=True, desc=object_name) as pbar:
            s3.download_file(
                bucket, 
                object_name, 
                file_path,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
            )
    except ClientError as e:
        logging.error(e)
        return False
    return True

def sync_to_s3(local_dir, bucket, prefix='', subdirectory=''):
    """Sync local directory or subdirectory to S3 bucket"""
    full_local_path = os.path.join(local_dir, subdirectory)
    if not os.path.exists(full_local_path):
        logging.error(f"Local directory does not exist: {full_local_path}")
        return

    for root, dirs, files in os.walk(full_local_path):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_dir)
            s3_path = os.path.join(prefix, relative_path).replace("\\", "/")
            
            if upload_file(local_path, bucket, s3_path):
                logging.info(f"Uploaded {local_path} to {s3_path}")
            else:
                logging.error(f"Failed to upload {local_path}")

def sync_from_s3(bucket, prefix, local_dir, subdirectory=''):
    """Sync S3 bucket to local directory or subdirectory"""
    full_prefix = os.path.join(prefix, subdirectory).replace("\\", "/")
    full_local_path = os.path.join(local_dir, subdirectory)

    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
        if 'Contents' in result:
            for obj in result['Contents']:
                s3_path = obj['Key']
                relative_path = os.path.relpath(s3_path, prefix)
                local_path = os.path.join(local_dir, relative_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                if download_file(bucket, s3_path, local_path):
                    logging.info(f"Downloaded {s3_path} to {local_path}")
                else:
                    logging.error(f"Failed to download {s3_path}")

def main():
    parser = argparse.ArgumentParser(description="Sync directories with S3")
    parser.add_argument('direction', choices=['to_s3', 'from_s3'], help="Sync direction")
    parser.add_argument('--dir', choices=['data', 'models'], required=True, help="Main directory to sync")
    parser.add_argument('--subdir', default='', help="Subdirectory to sync (optional)")
    args = parser.parse_args()

    if args.direction == 'to_s3':
        sync_to_s3(args.dir, BUCKET_NAME, args.dir, args.subdir)
    else:  # from_s3
        sync_from_s3(BUCKET_NAME, args.dir, args.dir, args.subdir)

if __name__ == "__main__":
    main()
