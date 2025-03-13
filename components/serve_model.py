from kfp import dsl

@dsl.component(
    base_image="python:3.9-slim",
    packages_to_install=["kserve>=0.13.0", "kubernetes"]
)
def serve_model(
    model_path: str,
    minio_endpoint: str,
    minio_access_key: str,
    minio_secret_key: str,
    minio_bucket: str
) -> None:
    from kserve import KServeClient, V1beta1InferenceService, V1beta1InferenceServiceSpec, V1beta1PredictorSpec
    from kubernetes import client
    from datetime import datetime

    namespace = "default"
    now = datetime.now()
    v = now.strftime("%Y-%m-%d--%H-%M-%S")
    name = f"sentiment-model-{v}"

    isvc = V1beta1InferenceService(
        api_version="serving.kserve.io/v1beta1",
        kind="InferenceService",
        metadata=client.V1ObjectMeta(name=name, namespace=namespace),
        spec=V1beta1InferenceServiceSpec(
            predictor=V1beta1PredictorSpec(
                service_account_name="default",  # Ensure MinIO access if needed
                containers=[
                    client.V1Container(
                        name="kserve-container",
                        image="yourusername/distilbert-serving:latest",
                        env=[
                            client.V1EnvVar(name="STORAGE_URI", value=model_path),
                            client.V1EnvVar(name="MINIO_ENDPOINT", value=minio_endpoint),
                            client.V1EnvVar(name="MINIO_ACCESS_KEY", value=minio_access_key),
                            client.V1EnvVar(name="MINIO_SECRET_KEY", value=minio_secret_key),
                        ]
                    )
                ]
            )
        )
    )

    KServe = KServeClient()
    KServe.create(isvc)
    print(f"Deployed InferenceService: {name}")