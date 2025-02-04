# ML Mini Platform 🚀

## 📖 Description of the Solution

The **ML Mini Platform** is my lightweight, containerized experimental solution designed to streamline the entire lifecycle of machine learning models. With this platform, I aim to:

- **Catalog Datasets** 📂: Easily upload and normalize dataset files while storing important metadata in a database.
- **Train Models** 🤖: Automatically train ML models using cataloged datasets, save model artifacts in object storage (MinIO or S3), and register them with versioning.
- **Promote Models** ⬆️: Seamlessly promote models between environments—**dev**, **staging**, and **production**—by creating new artifacts and updating the registry. *Note: A model can only be promoted (from dev to staging or production); demotion is not allowed.*
- **Serve Predictions** 🔮: Provide a dedicated endpoint to serve predictions using the latest model for a given dataset and environment.
- **Interactive API Documentation** 📜: Explore and test all endpoints through automatically generated Swagger UI.

This platform is an experimental project aimed at demonstrating a complete end-to-end machine learning workflow in a containerized environment.

---

## ✨ Features

- **Dataset Cataloging**  
  📂 Upload datasets via an API. The system automatically normalizes file names (e.g., converting `"My DataSet.csv"` to `"my_dataset"`), stores metadata in the database, and saves files in object storage.

- **Automated Model Training & Registration**  
  🤖 Train models using your cataloged datasets. The platform saves model artifacts with a consistent naming convention (including dataset, environment, and version) and registers them in a SQLite (or configurable) database.

- **Model Promotion**  
  ⬆️ Promote models between environments. The platform supports three environments:
  - **dev**: The initial environment where models are trained.
  - **staging**: An intermediate environment for further validation.
  - **production**: The final environment for serving predictions.
  
  Models can only be promoted (e.g., from dev to staging or production); demotion is not permitted.

- **Prediction Serving**  
  🔮 Retrieve predictions by specifying the dataset and environment. The system automatically loads the latest model version to serve predictions.

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
  - **Pydantic**: For request/response validation (with detailed examples)  
  - **Uvicorn**: ASGI server for FastAPI  
  - **Boto3**: For interacting with object storage  
  - **Pandas**: For dataset handling and preprocessing  
  - **Scikit-Learn (sklearn)**: For ML model training and preprocessing

---

## 🚀 How to Run (Using Docker Compose)

1. **Prerequisites:**  
   - Install [Docker](https://docs.docker.com/get-docker/)  
   - Install [Docker Compose](https://docs.docker.com/compose/install/)

2. **Build & Run:**  
   In your project directory, run:
   ```bash
   docker compose up --build
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
The Swagger UI provides interactive API documentation with detailed request and response models (e.g., `DatasetResponse`, `TrainResponse`, `PredictResponse`) that include example payloads.

---

## 📚 Test Solution Guide

The following steps outline an end-to-end workflow for testing the ML Mini Platform.

### 1. Upload a Sample Dataset

1. In the Swagger UI, navigate to the **Dataset** section.
2. Select the `POST /datasets/upload` endpoint.
3. Upload the sample file **Iris.csv** from the `samples` folder.  
   *(You can leave the default dataset name as `iris_dataset` or change it.)*
4. Click **Execute**.
5. Verify that the response includes details such as dataset ID, name, location, and creation timestamp.

---

### 2. Train a New Model

1. In Swagger, navigate to the **Models** section.
2. Select the `POST /train` endpoint.
3. Fill in the parameters:
   - **dataset_name**: Use the name of your uploaded dataset (e.g., `my_dataset`).
   - **target_column**: For example, `Species`.
   - **model**: Provide a sample configuration, for instance:
     ```json
     {
       "model_class": "RandomForestClassifier",
       "model_params": {
         "n_estimators": 100,
         "max_depth": 5
       }
     }
     ```
   - Optionally, adjust `test_size` and `random_state`.
4. Click **Execute**.
5. Confirm that the response (modeled by `TrainResponse`) includes details like the model file name, environment (default is `dev`), version, target column, features, and training metrics.

---

### 3. Promote the Model to Production

Our solution supports three environments: **dev**, **staging**, and **production**.  
- **dev**: The environment where models are initially trained.  
- **staging**: An intermediate environment for further validation.  
- **production**: The final environment used for serving predictions.

When you train a model, it is created in the **dev** environment. You can then promote the model to a higher environment (staging or production) using the `/promote` endpoint.  
*Remember: Once promoted, models cannot be demoted to a lower environment.*

1. In Swagger, select the `POST /promote` endpoint.
2. Provide:
   - **dataset_name**: Your dataset name.
   - **version**: The version number from the training step.
   - **environment**: Set this to `production` (or `staging` if you prefer an intermediate step).
3. Click **Execute**.
4. Verify that the response (modeled by `PromoteResponse`) confirms the promotion with the new model name, dataset, version, environment, and promotion timestamp.

---

### 4. Predict Using the Model

1. In Swagger, navigate to the `POST /predict` endpoint.
2. Provide the necessary parameters:
   - **dataset_name**: (e.g., `my_dataset`)
   - **environment**: (e.g., `production`)
   - **version**: (if applicable)
   - **features**: A list of dictionaries containing the input features. See the example in the Swagger UI.
3. Click **Execute**.
4. Verify that the response (modeled by `PredictResponse`) returns the expected list of predictions.

---

### Optional Testing Steps

If you’d like to explore further, the following optional endpoints can help you assess additional features:

#### List Datasets

- **Endpoint:** `GET /datasets`  
- **Purpose:** Retrieve all datasets in the catalog.  
- **Instructions:** Execute the endpoint in Swagger and review the dataset objects.

#### List Models

- **Endpoint:** `GET /models`  
- **Purpose:** Retrieve all registered models.  
- **Instructions:** Run the endpoint and check that the response includes details such as model name, version, environment, associated dataset, and timestamps.

#### List Models by Environment and by Dataset

- **By Environment:**  
  - **Endpoint:** `GET /models/by-environment`  
  - Provide an environment (e.g., `dev`) as a query parameter.
- **By Dataset:**  
  - **Endpoint:** `GET /models/by-dataset`  
  - Provide a dataset name (e.g., `my_dataset`) as a query parameter.
- **Instructions:** Execute these endpoints to verify that the filtering works correctly.

#### Remove a Model

- **Endpoint:** `DELETE /models/remove`  
- **Purpose:** Remove a specific model version and its associated file from storage.  
- **Instructions:** Provide the dataset name, model version, and environment. Execute the request and confirm that the response indicates successful removal.

#### Remove a Dataset

- **Endpoint:** `DELETE /datasets/{dataset_id}`  
- **Purpose:** Delete a dataset (only if no models are associated with it).  
- **Instructions:** Identify a dataset with no linked models using the list endpoint, then execute the delete request and verify the response.

---

## 🔮 Future Decisions to Improve the Platform

- **Web UI for Easier Management:**  
  💻 Introduce a user-friendly web interface using React or Vue.js for seamless dataset management, model tracking, and deployment—eliminating the need for manual API calls.

- **Expanded Model Support:**  
  📊 Extend the platform beyond Scikit-Learn by supporting additional ML and deep learning frameworks (e.g., TensorFlow, PyTorch, XGBoost) to enable a broader range of model training and deployment options.

- **Enhanced Database Management:**  
  🗄️ Migrate from SQLite to PostgreSQL or MySQL for improved performance, concurrency, and scalability.

- **Robust Model Registry:**  
  🏷️ Integrate a dedicated model registry solution (e.g., MLflow) to track experiments, hyperparameters, and model metrics in more detail.

- **CI/CD Integration:**  
  🔄 Automate testing, training, and deployment pipelines using GitHub Actions, GitLab CI/CD, or similar tools to streamline the promotion process.

- **Cloud Object Storage:**  
  ☁️ Replace or supplement MinIO with Amazon S3 or another cloud storage service for improved durability and scalability.

- **Monitoring & Logging:**  
  📈 Integrate centralized logging (ELK, CloudWatch) and monitoring (Prometheus, Grafana) to ensure high availability and rapid troubleshooting.

- **Security Enhancements:**  
  🔐 Add API authentication (JWT/OAuth2) and role-based access controls to secure endpoints and protect sensitive data.

- **Asynchronous Processing:**  
  ⚡ Add an asynchronous option using a queue and workers to handle heavy training tasks, improving performance during peak loads.

- **Support for More Model Classes:**  
  🧩 Enhance training capabilities by adding support for more model classes to accommodate a wider variety of ML and deep learning use cases.

- **Generative AI & Fine-Tuning:**  
  🧠 Introduce support for fine-tuning and training Generative AI models, catering to advanced AI applications.

- **Microservices Architecture:**  
  🔄 Transition to a microservices architecture to decouple components, improve scalability, and streamline updates.

---

## ☁️ AWS Architecture Overview

To make the ML Mini Platform more robust, reliable, and cost-effective, it can be expanded on AWS using the following architecture:

- **Container Orchestration:**  
  Use **Amazon EKS (Elastic Kubernetes Service)** or **Amazon ECS with Fargate** to deploy and manage containerized services.  
  - **EKS** offers Kubernetes-native management for scalability and fine-grained control.  
  - **ECS with Fargate** provides a serverless container solution that eliminates infrastructure management.  
  🛠️

- **Object Storage:**  
  Replace or supplement MinIO with **Amazon S3** for scalable, durable, and cost-effective storage of datasets and model artifacts.  
  ☁️📦

- **Managed Database:**  
  Migrate from SQLite to **Amazon RDS (PostgreSQL/MySQL)** for improved performance, concurrency, and scalability.  
  🗄️🔒

- **Data Processing:**  
  Use **Amazon EMR** for large-scale data preprocessing using Apache Spark or Hadoop, offloading intensive computations from the main application.  
  🔄📊

- **Machine Learning Services:**  
  Integrate **Amazon SageMaker** for comprehensive model training, tuning, and deployment. SageMaker offers managed environments for distributed training, hyperparameter optimization, and real-time inference.  
  🤖✨

- **Load Balancing & Auto Scaling:**  
  Deploy an **Elastic Load Balancer (ELB)** to distribute traffic across multiple containers or instances. Use **Auto Scaling** groups to automatically adjust capacity based on demand, ensuring high availability.  
  ⚖️📈

- **Monitoring & Logging:**  
  Integrate **Amazon CloudWatch** for real-time monitoring and logging, enabling automated alerts and rapid troubleshooting.  
  📈🛡️

- **Security & Access Management:**  
  Use **AWS IAM** and **Security Groups** to enforce strict access controls and network segmentation, ensuring only authorized access to sensitive data and services.  
  🔐✅

- **CI/CD Integration:**  
  Implement CI/CD pipelines using **AWS CodePipeline** and **AWS CodeDeploy** (or similar tools like GitHub Actions) to automate testing, training, and deployment workflows, reducing manual intervention.  
  🚀🔄

This AWS architecture provides a scalable, reliable, and cost-effective foundation to expand the ML Mini Platform, ensuring it can handle production workloads and grow with your needs.

---

## Conclusion

The ML Mini Platform offers an end-to-end experimental solution for managing the complete machine learning lifecycle—from dataset upload and model training to promotion and prediction. This README outlines the core features, provides detailed sample payloads, and includes a friendly test solution guide for interacting with the APIs via Swagger. Additionally, the Future Decisions and AWS Architecture sections demonstrate how the platform can be further enhanced and scaled for production environments.

I view this platform as a work-in-progress experiment, and I warmly welcome your feedback and suggestions. Your insights are invaluable in helping me refine and improve the solution.

Thank you for taking the time to review my project. Happy coding and exploring! 🚀🎉
