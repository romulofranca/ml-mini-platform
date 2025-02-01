# ML Mini Platform 🚀

## 📖 Description of the Solution

The **ML Mini Platform** is a lightweight, containerized solution that streamlines the complete lifecycle of machine learning models. It allows you to:

- **Catalog Datasets**: Upload and normalize dataset files, and store metadata in a database.  
- **Train Models**: Automatically train ML models on cataloged datasets, store model artifacts in object storage (MinIO or S3), and register them with versioning.  
- **Promote Models**: Easily promote models between environments (e.g., from _dev_ to _staging_ to _production_) by creating new artifacts and updating the registry.  
- **Serve Predictions**: Provide a dedicated endpoint for serving predictions from the latest model for a given dataset and environment.  
- **Interactive API Documentation**: Use the automatically generated Swagger UI to explore and test all endpoints.

---

## ✨ Features

- **Dataset Cataloging**  
  📂 Upload datasets via an API, automatically normalizing names (lowercase, underscores, no extension) and storing metadata in a database.

- **Automated Model Training & Registration**  
  🤖 Train models using cataloged datasets. The platform saves model artifacts with a consistent naming convention (including dataset, environment, and version) and registers them in a SQLite (or configurable) database.

- **Model Promotion**  
  ⬆️ Easily promote models from one environment to another (e.g., _dev_ → _staging_ → _production_), with new artifact names and updated registry entries.

- **Prediction Serving**  
  🔮 Retrieve predictions by specifying the dataset and environment, with the system automatically loading the latest model version.

- **Interactive Documentation**  
  📜 Access the Swagger UI for interactive API exploration and testing.

- **Containerization**  
  🐳 Uses Docker & Docker Compose for rapid, consistent deployment and scaling.

---

## 🛠️ Components

- **FastAPI Application**  
  Provides API endpoints for dataset cataloging, model training, promotion, and serving predictions.

- **Database (SQLite by default)**  
  Stores catalog entries for datasets and a model registry with global incremental versioning.

- **Object Storage (MinIO)**  
  Stores dataset files and model artifacts.

- **Docker & Docker Compose**  
  Containerizes the entire solution with separate services for the application, MinIO, and an initialization container for bucket creation.

- **Python Libraries**  
  - **SQLAlchemy**: ORM for database interactions  
  - **Pydantic**: For request/response validation  
  - **Uvicorn**: ASGI server for FastAPI  
  - **Boto3**: For interacting with object storage

---

## 🚀 How to Run (Using Docker Compose)

1. **Prerequisites:**  
   - Install [Docker](https://docs.docker.com/get-docker/)  
   - Install [Docker Compose](https://docs.docker.com/compose/install/)

2. **Build & Run:**  
   In your project directory, run:
   ```bash
   docker-compose up --build
   ```
   This command will:
   - Build the application container.
   - Start the MinIO service with persistent storage.
   - Run an initialization container that creates the `datasets` and `trained-models` buckets.
   - Start the FastAPI application container.

3. **Database Initialization:**  
   The SQLite database (`ml_registry.db`) is stored in the mounted volume `/app/data` and is automatically created (if not existing) when the app starts.

---

## 🌐 How to Access and Use

### Swagger UI Documentation

Once the platform is running, open your browser and navigate to:  
[http://localhost:8000/docs](http://localhost:8000/docs)  
Here, you can interact with all API endpoints.

### Sample Payloads

#### 1. Upload Dataset
**Endpoint:** `POST /datasets/upload`  
**Payload (form-data):**  
- **file:** Choose a file (e.g., `My DataSet.csv`)  
  
The system normalizes the name to `my_dataset` and creates a catalog entry.

#### 2. Train Model
**Endpoint:** `POST /train`  
**Payload (JSON):**
```json
{
  "dataset_name": "my_dataset",
  "use_example": false,
  "model": {
    "module": "ensemble",
    "class": "RandomForestClassifier",
    "params": {
      "n_estimators": 100
    }
  },
  "target_column": "target",
  "test_size": 0.2,
  "random_state": 42
}
```
This request:
- Loads the dataset (using the catalog entry for `my_dataset`),
- Trains a model,
- Saves the artifact (e.g., `my_dataset_model_dev_v1.pkl`),
- Registers the model in the registry with version `1`.

#### 3. Promote Model
**Endpoint:** `POST /promote`  
**Payload (JSON):**
```json
{
  "dataset_name": "my_dataset",
  "version": 1,
  "target_stage": "staging"
}
```
This request promotes the model version `1` from `dev` to `staging`, creating a new artifact (e.g., `my_dataset_model_staging_v1.pkl`) and updating the registry.

#### 4. Predict
**Endpoint:** `POST /predict`  
**Payload (JSON):**
```json
{
  "dataset_name": "my_dataset",
  "environment": "staging",
  "input_data": {
    "features": [
      [5.1, 3.5, 1.4, 0.2],
      [6.2, 3.4, 5.4, 2.3]
    ]
  }
}
```
This request uses the latest model for `my_dataset` in the staging environment to return predictions.

---

## 🔄 Workflow: Dataset → Train → Promote

1. **Dataset Upload:**  
   - The user uploads a dataset file through the `/datasets/upload` endpoint.
   - The file name is normalized (e.g., `"My DataSet.csv"` → `"my_dataset"`).
   - A catalog entry is created in the **DatasetCatalog** table in SQLite.
   - The file is stored in the MinIO bucket for datasets.

2. **Model Training:**  
   - The user sends a payload to `/train` with the dataset name and model configuration.
   - The application loads the dataset (from MinIO or local storage as per the catalog).
   - A model is trained using the provided configuration.
   - The model artifact is stored in the MinIO bucket for trained models using a naming convention like `my_dataset_model_dev_v1.pkl`.
   - An entry is created in the **ModelRegistry** table with details (version, environment, metrics, parameters).

3. **Model Promotion:**  
   - After testing, the user promotes a model by calling `/promote`.
   - The system creates a new artifact (e.g., `my_dataset_model_staging_v1.pkl`) and updates the registry entry to reflect the new environment (promotion timestamp is set).

4. **Database Appearance:**  
   - **DatasetCatalog Table:** Contains rows with dataset names (e.g., `my_dataset`), descriptions, file locations, and timestamps.  
   - **ModelRegistry Table:** Contains rows with fields like:
     - `dataset_id` (linking to the catalog entry)
     - `version` (global incremental version)
     - `stage` (e.g., `dev`, `staging`, `production`)
     - `artifact_path` (name of the saved model file)
     - `metrics`, `parameters` (stored as JSON strings)
     - Timestamps for training and promotion.

---

## 🔮 Future Decisions to Improve the Platform

- **Enhanced Database Management:**  
  Migrate from SQLite to PostgreSQL or MySQL for better performance, concurrency, and scalability.

- **Robust Model Registry:**  
  Integrate a dedicated model registry solution (e.g., MLflow) to track experiments, hyperparameters, and model metrics in more detail.

- **CI/CD Integration:**  
  Automate testing, training, and deployment pipelines (using GitHub Actions, GitLab CI/CD, etc.) to streamline the promotion process.

- **Cloud Object Storage:**  
  Replace or supplement MinIO with Amazon S3 or another cloud storage service for improved durability and scalability.

- **Monitoring & Logging:**  
  Integrate centralized logging (ELK, CloudWatch) and monitoring (Prometheus, Grafana) for high availability and rapid troubleshooting.

- **Security Enhancements:**  
  Add API authentication (JWT/OAuth2) and role-based access controls to secure the endpoints.

---

## ☁️ Architecture on AWS

**AWS Architecture Overview:**

This platform can be deployed on AWS using the following components to ensure reliability, scalability, and cost-effectiveness:

- **Amazon ECS/Fargate or EC2:**  
  Run containerized services (FastAPI application, MinIO replaced by S3 in production) using ECS/Fargate for serverless containers or EC2 with auto-scaling groups.

- **Amazon S3:**  
  Replace MinIO for production-grade object storage. S3 offers high durability, scalability, and low cost for storing datasets and model artifacts.

- **Amazon RDS (PostgreSQL/MySQL):**  
  Replace SQLite with a managed relational database for higher concurrency, availability, and ease of scaling.

- **Elastic Load Balancer (ELB):**  
  Distribute incoming traffic across multiple instances of the ML platform to ensure high availability.

- **AWS CloudWatch:**  
  Monitor logs and performance metrics, and set up alarms for operational insights.

- **AWS IAM & Security Groups:**  
  Enforce least-privilege access policies and secure network communications between services.

- **AWS CodePipeline/CodeDeploy:**  
  Automate the CI/CD workflow to deploy updates reliably and quickly.

This AWS architecture is designed to be highly reliable, scalable, and cost-effective, ensuring that your ML platform can handle production workloads while keeping operational costs low.
