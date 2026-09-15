from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from feature_engineering import engineer_features


# Project paths
DATA_PATH = Path("data/APL_Logistics.csv")
MODEL_PATH = Path("models/late_delivery_model.pkl")
REPORTS_DIR = Path("reports")
OUTPUT_PATH = REPORTS_DIR / "SHAP_Summary.png"
SAMPLE_SIZE = 1_000


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH.resolve()}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH.resolve()}")

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    print("Dataset shape:", df.shape)

    print("Creating features...")
    df = engineer_features(df)

    print("Loading trained model...")
    bundle = joblib.load(MODEL_PATH)

    # The saved object is expected to contain these keys.
    model = bundle["model"]
    features = bundle["features"]
    model_name = str(bundle.get("model_name", "Unknown"))

    print("Best model:", model_name)
    print("Pipeline steps:", list(model.named_steps.keys()))

    missing_features = [column for column in features if column not in df.columns]
    if missing_features:
        raise KeyError(
            "The following trained features are missing from the dataset after "
            f"feature engineering: {missing_features}"
        )

    X = df[features].copy()

    sample_size = min(SAMPLE_SIZE, len(X))
    print(f"Selecting {sample_size} sample rows...")
    X_sample = X.sample(sample_size, random_state=42)

    # Support either naming convention:
    #   preprocessor + model
    #   preprocess + classifier
    preprocessor = model.named_steps.get("preprocessor")
    if preprocessor is None:
        preprocessor = model.named_steps.get("preprocess")

    trained_model = model.named_steps.get("model")
    if trained_model is None:
        trained_model = model.named_steps.get("classifier")

    if preprocessor is None or trained_model is None:
        raise KeyError(
            "Could not find preprocessing/model steps. Expected either "
            "('preprocessor', 'model') or ('preprocess', 'classifier'). "
            f"Found: {list(model.named_steps.keys())}"
        )

    print("Transforming data...")
    X_transformed = preprocessor.transform(X_sample)

    # SHAP and plotting are more reliable with dense arrays for ordinary-sized
    # samples. The selected sample is limited to 1,000 rows to control memory.
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    # Obtain names after one-hot encoding, if supported.
    try:
        feature_names = preprocessor.get_feature_names_out()
    except (AttributeError, TypeError):
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    feature_names = [str(name) for name in feature_names]

    print("Creating SHAP values...")
    model_name_lower = model_name.lower()
    is_tree_model = any(
        name in model_name_lower
        for name in ["xgboost", "xgb", "random forest", "randomforest", "lightgbm"]
    )

    if is_tree_model:
        explainer = shap.TreeExplainer(trained_model)
        shap_result = explainer.shap_values(X_transformed)
    else:
        # LinearExplainer is appropriate for LogisticRegression and linear models.
        explainer = shap.LinearExplainer(trained_model, X_transformed)
        shap_result = explainer.shap_values(X_transformed)

    # Older SHAP versions can return a list for binary classification.
    if isinstance(shap_result, list):
        shap_values = shap_result[1] if len(shap_result) == 2 else shap_result[0]
    elif hasattr(shap_result, "values"):
        shap_values = shap_result.values
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    else:
        shap_values = shap_result

    print("Creating SHAP graph...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print("SHAP analysis completed successfully.")
    print("Graph saved at:")
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
