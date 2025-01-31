# ML Mini Platform

ML Mini Platform is a generic machine learning training and prediction service built with FastAPI.  
It allows data analysts to upload datasets, train models with custom configurations, and use the trained models for prediction.  
Trained models (and their metadata) as well as datasets are stored in an S3-compatible storage (MinIO).

## Endpoints

### 1. **Health Check**

- **URL:** `/`  
- **Method:** `GET`  
- **Description:** Checks the status of the API.  
- **Response Example:**
  ```json
  {
    "message": "ML Mini Platform is running!"
  }
  ```

### 2. **Upload Dataset**

- **URL:** `/upload`  
- **Method:** `POST`  
- **Description:** Upload a dataset file to the storage.  
- **Request:**  
  Use multipart/form-data with a file field.
- **Example using `curl`:**
  ```bash
  curl -F "file=@/path/to/dataset.csv" http://localhost:8000/upload
  ```
- **Response Example:**
  ```json
  {
    "message": "File 'dataset.csv' uploaded successfully!"
  }
  ```

### 3. **List Datasets**

- **URL:** `/list-datasets`  
- **Method:** `GET`  
- **Description:** Lists all dataset files available in storage.
- **Response Example:**
  ```json
  {
    "datasets": [
      "dataset.csv",
      "CO2_emission.csv"
    ]
  }
  ```

### 4. **List Trained Models**

- **URL:** `/list-models`  
- **Method:** `GET`  
- **Description:** Lists all trained models stored in storage.
- **Response Example:**
  ```json
  {
    "models": [
      "CO2_emission.csv/version-0b930a28-22bb-4f8e-9c30-48650a83545a/model.pkl",
      "iris-sample/version-1234/model.pkl"
    ]
  }
  ```

### 5. **Train Model**

- **URL:** `/train`  
- **Method:** `POST`  
- **Description:** Trains a machine learning model based on a provided configuration.  
  During training, the feature names are automatically stored as part of the model metadata so that predictions can later be made without manually specifying the schema.
- **Request Payload Example:**
  ```json
  {
    "dataset_path": "CO2_emission.csv",
    "use_example": false,
    "model": {
      "module": "ensemble",
      "class": "RandomForestClassifier",
      "params": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
      }
    },
    "target_column": "label",
    "test_size": 0.3,
    "random_state": 42
  }
  ```
  **Note:**  
  - Set `"use_example": true` to use the built-in Iris dataset.  
  - When using your custom dataset, ensure that the file has already been uploaded to the storage.

- **Response Example:**
  ```json
  {
    "message": "Model trained successfully",
    "metrics": {
      "1": { "precision": 0.909, "recall": 0.5, "f1-score": 0.645, "support": 20 },
      "3": { "precision": 0.814, "recall": 0.479, "f1-score": 0.603, "support": 73 },
      "5": { "precision": 0.547, "recall": 0.837, "f1-score": 0.661, "support": 98 },
      "6": { "precision": 0.75, "recall": 0.081, "f1-score": 0.146, "support": 37 },
      "7": { "precision": 0.603, "recall": 0.83, "f1-score": 0.698, "support": 53 },
      "accuracy": 0.619,
      "macro avg": { "precision": 0.724, "recall": 0.545, "f1-score": 0.551, "support": 281 },
      "weighted avg": { "precision": 0.679, "recall": 0.619, "f1-score": 0.584, "support": 281 },
      "f1_score": 0.584
    },
    "model_path": "CO2_emission.csv/version-0b930a28-22bb-4f8e-9c30-48650a83545a/model.pkl"
  }
  ```

### 6. **Predict**

- **URL:** `/predict`  
- **Method:** `POST`  
- **Description:** Uses a trained model to make predictions on new input data.  
  The API automatically retrieves the stored feature names from the model's metadata and converts the raw input into a DataFrame.
- **Request Payload Example:**
  ```json
  {
    "model_path": "CO2_emission.csv/version-0b930a28-22bb-4f8e-9c30-48650a83545a/model.pkl",
    "input_data": {
      "features": [
        [2.5, 6, 15.0, 20.0, 25.0, 300.0, 0.5, 1.2, 3.4, 5.6, 7.8]
      ]
    }
  }
  ```
  **Note:**  
  The list of feature values must match the number of features used during training. The API automatically handles conversion based on stored metadata.
  
- **Response Example:**
  ```json
  {
    "predictions": [3]
  }
  ```

### 7. **OpenAPI Schema**

- **URL:** `/openapi.json`  
- **Method:** `GET`  
- **Description:** Retrieves the OpenAPI schema for this API.

---

## Running the Application with Docker Compose

This project uses Docker Compose to run both the FastAPI application and MinIO (an S3-compatible storage service).

### Prerequisites

- [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) must be installed on your system.

### Steps to Run

1. **Build and Start the Services:**

   In your project directory (where `docker-compose.yml` is located), run:
   ```bash
   docker compose up --build
   ```

2. **Access the Services:**

   - **FastAPI Application:**  
     The API will be available at [http://localhost:8000](http://localhost:8000).  
     Swagger documentation is accessible at [http://localhost:8000/docs](http://localhost:8000/docs).

   - **MinIO:**  
     MinIO’s web interface will be available at [http://localhost:9000](http://localhost:9000).  
     Use the credentials `minioadmin` for both the username and password.  
     Create the buckets named `datasets` and `models` (or use the bucket names defined in your environment variables) via the web UI if they do not already exist.

3. **Testing the API:**

   You can now use tools like [Postman](https://www.postman.com/) or `curl` to interact with the endpoints as described above.

---

## Summary

- **Upload your dataset** using the `/upload` endpoint.
- **Train a model** using `/train` (either with your custom dataset or using the example Iris dataset).
- **Make predictions** using `/predict` without having to specify feature names manually—the platform handles this automatically by storing and retrieving metadata.
- **Run the entire stack** with Docker Compose which includes both the FastAPI application and MinIO.

