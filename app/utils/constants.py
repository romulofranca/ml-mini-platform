from enum import Enum


class EnvironmentEnum(str, Enum):
    """Global Enum for environment stages used across the application."""

    dev = "dev"
    staging = "staging"
    production = "production"


MODEL_MAPPING = {
    "KMeans": "sklearn.cluster",
    "DBSCAN": "sklearn.cluster",
    "IsolationForest": "sklearn.ensemble",
    "LinearRegression": "sklearn.linear_model",
    "DecisionTreeClassifier": "sklearn.tree",
}
