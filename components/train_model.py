from kfp import dsl
from typing import NamedTuple

class TrainModelOutputs(NamedTuple):
    model_path: str
    metrics_plot_path: str

@dsl.component(
    base_image="python:3.9-slim",
    packages_to_install=["torch", "transformers", "datasets", "scikit-learn", "matplotlib", "boto3"]
)
def train_model(
    train_data_path: str,
    val_data_path: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str
) -> TrainModelOutputs:
    import torch
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
    import pickle
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import matplotlib.pyplot as plt
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

    # Download data from MinIO
    s3.download_file(minio_bucket, "data/train_data.pkl", "/tmp/train_data.pkl")
    s3.download_file(minio_bucket, "data/val_data.pkl", "/tmp/val_data.pkl")
    with open("/tmp/train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open("/tmp/val_data.pkl", "rb") as f:
        val_data = pickle.load(f)

    # Tokenize
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_function(texts):
        return tokenizer(texts, padding="max_length", truncation=True, return_tensors="pt")

    train_encodings = tokenize_function(train_data["texts"])
    val_encodings = tokenize_function(val_data["texts"])

    class IMDbDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = IMDbDataset(train_encodings, train_data["labels"])
    val_dataset = IMDbDataset(val_encodings, val_data["labels"])

    # Model
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Training args
    training_args = TrainingArguments(
        output_dir="/output",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    )

    # Evaluation
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train and evaluate
    trainer.train()
    eval_results = trainer.evaluate()

    # Save model
    model.save_pretrained("/output/model")
    tokenizer.save_pretrained("/output/model")
    import tarfile
    with tarfile.open("/tmp/model.tar.gz", "w:gz") as tar:
        tar.add("/output/model", arcname="model")
    s3.upload_file("/tmp/model.tar.gz", minio_bucket, "model/model.tar.gz")

    # Visualization
    metrics = ["accuracy", "precision", "recall", "f1"]
    values = [eval_results[f"eval_{m}"] for m in metrics]
    plt.bar(metrics, values)
    plt.title("Model Evaluation Metrics")
    plt.savefig("/output/evaluation_metrics.png")
    s3.upload_file("/output/evaluation_metrics.png", minio_bucket, "metrics/evaluation_metrics.png")

    return TrainModelOutputs(
        model_path=f"s3://{minio_bucket}/model/model.tar.gz",
        metrics_plot_path=f"s3://{minio_bucket}/metrics/evaluation_metrics.png"
    )