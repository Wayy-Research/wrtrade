import click
import glob
import os
from google.cloud import storage

def upload_local_directory_to_gcs(local_path, bucket, gcs_path):
    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + '/**'):
        if not os.path.isfile(local_file):
           upload_local_directory_to_gcs(local_file, bucket, gcs_path + "/" + os.path.basename(local_file))
        else:
           remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
           blob = bucket.blob(remote_path)
           blob.upload_from_filename(local_file)

@click.group()
def wrtrade():
    pass

@wrtrade.command()
@click.argument('dirname')
def deploy(dirname):
    assert os.path.isdir(dirname)
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('wrtrade')
    upload_local_directory_to_gcs(dirname, bucket, dirname)

if __name__ == '__main__':
    deploy()