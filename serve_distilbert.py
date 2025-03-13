from flask import Flask, request, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import os
import boto3
from botocore.client import Config

app = Flask(__name__)

# MinIO credentials from environment (passed by KServe)
minio_endpoint = os.getenv("MINIO_ENDPOINT", "minio.minio.svc.cluster.local:9000")
minio_access_key = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
minio_secret_key = os.getenv("MINIO_SECRET_KEY", "minioadmin")
model_uri = os.getenv("STORAGE_URI", "s3://sentiment-bucket/model/model.tar.gz")

# Download model from MinIO
s3 = boto3.client(
    "s3",
    endpoint_url=f"http://{minio_endpoint}",
    aws_access_key_id=minio_access_key,
    aws_secret_access_key=minio_secret_key,
    config=Config(signature_version="s3v4")
)
model_tar = "/tmp/model.tar.gz"
model_dir = "/mnt/model"
os.makedirs(model_dir, exist_ok=True)
bucket, key = model_uri.replace("s3://", "").split("/", 1)
s3.download_file(bucket, key, model_tar)
import tarfile
with tarfile.open(model_tar, "r:gz") as tar:
    tar.extractall(model_dir)

tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
model = DistilBertForSequenceClassification.from_pretrained(model_dir)
model.eval()

@app.route("/v1/models/sentiment:predict", methods=["POST"])
def predict():
    data = request.json
    text = data.get("instances", [{}])[0].get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()[0]
    return jsonify({"predictions": [{"positive": probs[1], "negative": probs[0]}]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)