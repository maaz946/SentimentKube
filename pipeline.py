from kfp import dsl
from components.data_preparation import data_preparation, DataPreparationOutputs
from components.train_model import train_model, TrainModelOutputs
from components.serve_model import serve_model

@dsl.pipeline(
    name="Sentiment Analysis Pipeline",
    description="Trains and serves a sentiment analysis model with MinIO and KServe"
)
def sentiment_pipeline(
    minio_endpoint: str = "minio.minio.svc.cluster.local:9000",
    minio_access_key: str = "minioadmin",
    minio_secret_key: str = "minioadmin",
    minio_bucket: str = "sentiment-bucket"
):
    data_task = data_preparation(
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_bucket=minio_bucket
    )

    train_task = train_model(
        train_data_path=data_task.outputs["train_data_path"],
        val_data_path=data_task.outputs["val_data_path"],
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_bucket=minio_bucket
    ).after(data_task)

    serve_task = serve_model(
        model_path=train_task.outputs["model_path"],
        minio_endpoint=minio_endpoint,
        minio_access_key=minio_access_key,
        minio_secret_key=minio_secret_key,
        minio_bucket=minio_bucket
    ).after(train_task)

if __name__ == "__main__":
    from kfp.compiler import Compiler
    Compiler().compile(sentiment_pipeline, "sentiment_pipeline.yaml")
    client = dsl.Client(host="http://localhost:8080")
    client.create_run_from_pipeline_package("sentiment_pipeline.yaml", arguments={})