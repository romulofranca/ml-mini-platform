import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger(__name__)


def detect_target_column(df: pd.DataFrame) -> str:
    """
    Detect the target column from a DataFrame using possible names.
    """
    possible_target_columns = ["target", "label", "class", "y"]
    for col in possible_target_columns:
        if col in df.columns:
            return col
    return df.columns[-1]


def create_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Create a preprocessor for numeric and categorical columns.
    """
    numeric_features = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()
    categorical_features = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    logger.info("Preprocessor created successfully.")
    return preprocessor
