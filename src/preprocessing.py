
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(file_path):
    """
    Load the APL Logistics dataset.

    The dataset contains special characters, so latin1 encoding
    is used instead of the default UTF-8.
    """
    df = pd.read_csv(file_path, encoding="latin1")
    return df


def build_preprocessor(numerical_features, categorical_features):
    """
    Create the preprocessing pipeline.

    Numerical columns:
        - Fill missing values with median
        - Standardize using StandardScaler

    Categorical columns:
        - Fill missing values with most frequent value
        - Convert categories into numerical values using OneHotEncoder
    """

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ],
        remainder="drop"
    )

    return preprocessor


def get_feature_types(df, target_column):
    """
    Automatically identify numerical and categorical columns.

    The target column is excluded from the predictors.
    """

    feature_df = df.drop(columns=[target_column])

    numerical_features = feature_df.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = feature_df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return numerical_features, categorical_features


if __name__ == "__main__":

    DATA_PATH = "data/APL_Logistics.csv"
    TARGET = "Late_delivery_risk"

    df = load_data(DATA_PATH)

    print("=" * 60)
    print("APL LOGISTICS DATASET")
    print("=" * 60)

    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    numerical_features, categorical_features = get_feature_types(
        df,
        TARGET
    )

    print("\nNumerical Features:")
    for feature in numerical_features:
        print(f"  - {feature}")

    print("\nCategorical Features:")
    for feature in categorical_features:
        print(f"  - {feature}")

    print("\nTarget:")
    print(f"  - {TARGET}")

    print("\nPreprocessing pipeline created successfully.")

