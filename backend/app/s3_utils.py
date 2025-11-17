import os, boto3, botocore

def maybe_download_from_s3(local_path: str) -> str:
    bucket = os.getenv("S3_BUCKET")
    key = os.getenv("S3_KEY")
    region = os.getenv("AWS_REGION", "us-east-1")

    if not bucket or not key:
        return local_path  # nothing to fetch

    s3 = boto3.client("s3", region_name=region)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        s3.download_file(bucket, key, local_path)
        return local_path
    except botocore.exceptions.ClientError as e:
        raise RuntimeError(f"Failed to download model from s3://{bucket}/{key}: {e}")
