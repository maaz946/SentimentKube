from kfp import dsl
from typing import NamedTuple

# Define Outputs class outside the function
class DataPreparationOutputs(NamedTuple):
    train_data_path: str
    val_data_path: str

@dsl.component(base_image="python:3.9-slim", packages_to_install=["datasets", "boto3"])
def data_preparation(
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str
) -> DataPreparationOutputs:
    import os
    import pickle
    from datasets import load_dataset
    from sklearn.model_selection import train_test_split
    import boto3
    from botocore.client import Config

    # MinIO client
    s3 = boto3.client(
        "s3",
        endpoint_url=f"http://{minio_endpoint}",
        aws_access_key_id=minio_access_key,
        aws_secret_access_key=minio_secret_key,
        config=Config(signature_version="s3v4")
    )

    # Load and split IMDb dataset
    dataset = load_dataset("imdb", split="train[:5%]")
    reviews = dataset["text"]
    labels = dataset["label"]
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        reviews, labels, test_size=0.2, random_state=42
    )

    # Save to local files
    os.makedirs("/tmp", exist_ok=True)
    train_path = "/tmp/train_data.pkl"
    val_path = "/tmp/val_data.pkl"
    with open(train_path, "wb") as f:
        pickle.dump({"texts": train_texts, "labels": train_labels}, f)
    with open(val_path, "wb") as f:
        pickle.dump({"texts": val_texts, "labels": val_labels}, f)

    # Upload to MinIO
    s3.upload_file(train_path, minio_bucket, "data/train_data.pkl")
    s3.upload_file(val_path, minio_bucket, "data/val_data.pkl")

    # Return as NamedTuple instance
    return DataPreparationOutputs(
        train_data_path=f"s3://{minio_bucket}/data/train_data.pkl",
        val_data_path=f"s3://{minio_bucket}/data/val_data.pkl"
    )
