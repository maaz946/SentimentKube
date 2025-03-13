Sentiment Analysis Project Overview
===================================
This project implements a real-time sentiment analysis system using Kubeflow Pipelines (KFP v2), MinIO for storage, KServe for model serving, and a Flask web interface for testing. It runs locally on a Mac M1 `kind` cluster, avoiding cloud costs. Below is a description of each file in the `sentiment_project/` directory and its role in the workflow.

Directory Structure
-------------------
sentiment_project/
├── components/                # Python components for the KFP pipeline
│   ├── data_preparation.py    # Prepares and splits dataset, uploads to MinIO
│   ├── train_model.py         # Trains the DistilBERT model, saves to MinIO
│   ├── serve_model.py         # Deploys the trained model via KServe
├── Dockerfile.train           # Dockerfile for building the training container
├── Dockerfile.serve           # Dockerfile for building the serving container
├── inference_service.yaml     # Optional KServe InferenceService YAML (manual deployment)
├── pipeline.py                # Defines and runs the KFP pipeline
├── serve_distilbert.py        # KServe inference script for model prediction
├── web_app.py                 # Flask web app for testing predictions
└── templates/                 # HTML templates for the web app
    └── index.html             # Web interface template

File Descriptions
-----------------

1. components/data_preparation.py
   - Purpose: Prepares the IMDb dataset for training.
   - Functionality:
     - Loads a subset (5%) of the IMDb dataset using the `datasets` library.
     - Splits data into training (80%) and validation (20%) sets using `sklearn.model_split.train_test_split`.
     - Saves the split data as pickle files (`train_data.pkl`, `val_data.pkl`).
     - Uploads these files to MinIO under the `data/` prefix in the specified bucket (e.g., `sentiment-bucket`).
     - Returns: A `NamedTuple` with `train_data_path` and `val_data_path` (S3 URIs).
   - Dependencies: `datasets`, `boto3` (installed via `@dsl.component`).
   - Usage: First step in the pipeline, called by `pipeline.py`.

2. components/train_model.py
   - Purpose: Trains a DistilBERT model for sentiment analysis.
   - Functionality:
     - Downloads training and validation data from MinIO using paths from `data_preparation`.
     - Tokenizes text using `transformers.DistilBertTokenizer`.
     - Trains a `DistilBertForSequenceClassification` model with PyTorch and `Trainer`.
     - Evaluates the model, generating accuracy, precision, recall, and F1 metrics.
     - Creates a bar chart visualization of metrics and saves it as `evaluation_metrics.png`.
     - Saves the trained model as a tarball (`model.tar.gz`) and uploads it to MinIO under `model/`.
     - Uploads the metrics plot to MinIO under `metrics/`.
     - Returns: A `NamedTuple` with `model_path` and `metrics_plot_path` (S3 URIs).
   - Dependencies: `torch`, `transformers`, `datasets`, `scikit-learn`, `matplotlib`, `boto3`.
   - Usage: Second step in the pipeline, depends on `data_preparation`.

3. components/serve_model.py
   - Purpose: Deploys the trained model to KServe for inference.
   - Functionality:
     - Creates a KServe `InferenceService` resource using the model path from `train_model`.
     - Configures the service to use a custom container (`yourusername/distilbert-serving:latest`).
     - Passes MinIO credentials and model URI via environment variables.
     - Deploys the service in the `default` namespace with a timestamped name (e.g., `sentiment-model-2025-03-13--12-34-56`).
     - Prints the deployed service name for reference.
   - Dependencies: `kserve>=0.13.0`, `kubernetes`.
   - Usage: Final step in the pipeline, depends on `train_model`.

4. Dockerfile.train
   - Purpose: Defines the container image for training components.
   - Functionality:
     - Uses `python:3.9-slim` as the base image.
     - Installs required packages: `torch`, `transformers`, `datasets`, `scikit-learn`, `matplotlib`, `boto3`.
     - Sets working directory to `/app` (though not directly used since components run Python code).
   - Build Command: `docker buildx build --platform linux/arm64 -t yourusername/distilbert-train:latest -f Dockerfile.train .`
   - Usage: Implicitly used by `data_preparation` and `train_model` components via `@dsl.component`.

5. Dockerfile.serve
   - Purpose: Defines the container image for the KServe inference service.
   - Functionality:
     - Uses `python:3.9-slim` as the base image.
     - Installs `flask`, `torch`, `transformers`, `boto3` for serving and MinIO access.
     - Copies `serve_distilbert.py` into `/app` and sets it as the entrypoint.
   - Build Command: `docker buildx build --platform linux/arm64 -t yourusername/distilbert-serving:latest -f Dockerfile.serve .`
   - Usage: Used by `serve_model` to deploy the KServe InferenceService.

6. inference_service.yaml
   - Purpose: Optional manual KServe deployment configuration.
   - Functionality:
     - Defines an `InferenceService` named `sentiment-model` in the `default` namespace.
     - Specifies the custom container `yourusername/distilbert-serving:latest`.
     - Passes MinIO credentials and model URI via environment variables.
   - Usage: Apply manually with `kubectl apply -f inference_service.yaml` if not using `serve_model.py`.
   - Note: Typically redundant since `serve_model.py` handles deployment dynamically.

7. pipeline.py
   - Purpose: Orchestrates the entire ML workflow using KFP v2.
   - Functionality:
     - Defines a pipeline with three steps: `data_preparation`, `train_model`, and `serve_model`.
     - Sets dependencies: `train_model` runs after `data_preparation`, `serve_model` after `train_model`.
     - Passes MinIO configuration (endpoint, credentials, bucket) as parameters.
     - Compiles the pipeline to `sentiment_pipeline.yaml` and runs it via the KFP client.
   - Run Command: `python pipeline.py`
   - Usage: Main entrypoint to execute the pipeline, monitor at `http://localhost:8080`.

8. serve_distilbert.py
   - Purpose: Implements the inference logic for KServe.
   - Functionality:
     - Runs a Flask app exposing a `/v1/models/sentiment:predict` endpoint.
     - Downloads the model from MinIO using credentials from environment variables (set by `serve_model`).
     - Loads the DistilBERT model and tokenizer from the downloaded tarball.
     - Predicts sentiment (positive/negative probabilities) for input text from POST requests.
     - Returns predictions in KServe-compatible JSON format.
   - Dependencies: `flask`, `torch`, `transformers`, `boto3` (installed via `Dockerfile.serve`).
   - Usage: Runs inside the KServe container deployed by `serve_model`.

9. web_app.py
   - Purpose: Provides a user interface to test the deployed model.
   - Functionality:
     - Runs a Flask app with a single route (`/`) for GET (display form) and POST (submit review).
     - Sends review text to the KServe endpoint (`MODEL_URL`) and displays the sentiment prediction.
     - Handles errors (e.g., empty input, failed requests) with user feedback.
   - Dependencies: `flask`, `requests` (must be installed locally: `pip install flask requests`).
   - Run Command: `python web_app.py`
   - Usage: Access at `http://localhost:5000` to test predictions.

10. templates/index.html
    - Purpose: HTML template for the web interface.
    - Functionality:
      - Displays a textarea for entering a review and a submit button.
      - Shows the review, predicted sentiment, and probabilities (positive/negative) after submission.
      - Displays error messages in red if prediction fails or input is invalid.
    - Usage: Rendered by `web_app.py` using Flask’s `render_template`.

Workflow
--------
1. Prerequisites:
   - Install MinIO (`kubectl apply -f minio_setup.yaml`), KServe, and Kubeflow per docs.
   - Create MinIO bucket (`sentiment-bucket`) via UI or `mc mb local/sentiment-bucket`.
2. Build Images:
   - `docker buildx build --platform linux/arm64 -t yourusername/distilbert-train:latest -f Dockerfile.train .`
   - `docker buildx build --platform linux/arm64 -t yourusername/distilbert-serving:latest -f Dockerfile.serve .`
   - Load to kind: `kind load docker-image yourusername/distilbert-train:latest --name sentiment-analysis`, repeat for `distilbert-serving`.
3. Run Pipeline:
   - `python pipeline.py` (port-forward KFP UI: `kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80 &`).
4. Test:
   - Update `MODEL_URL` in `web_app.py` with the KServe URL (e.g., `kubectl get inferenceservice -n default`).
   - `python web_app.py`, visit `http://localhost:5000`.

Notes
-----
- Replace `yourusername` with your Docker Hub username in image names.
- Ensure MinIO is running and accessible within the cluster.
- The pipeline uses KFP v2’s `@dsl.component` for modularity and modern type hints.
