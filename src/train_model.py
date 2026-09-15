
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from feature_engineering import engineer_features, get_model_features


# ---------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------

DATA_PATH = "data/APL_Logistics.csv"
MODEL_PATH = "models/late_delivery_model.pkl"

TARGET = "Late_delivery_risk"


# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------

print("=" * 60)
print("APL LOGISTICS - LATE DELIVERY PREDICTION")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH,
    encoding="latin1"
)

print("Dataset shape:", df.shape)


# ---------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------

print("\nCreating business features...")

df = engineer_features(df)

features = get_model_features(df)

X = df[features]
y = df[TARGET]

print("Number of features:", len(features))


# ---------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ---------------------------------------------------------

print("\nSplitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows :", len(X_test))


# ---------------------------------------------------------
# 5. IDENTIFY DATA TYPES
# ---------------------------------------------------------

numerical_features = X_train.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()


# ---------------------------------------------------------
# 6. PREPROCESSING
# ---------------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
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
    ]
)


# ---------------------------------------------------------
# 7. LOGISTIC REGRESSION
# ---------------------------------------------------------

print("\nTraining Logistic Regression...")

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

logistic_model.fit(X_train, y_train)

logistic_probability = logistic_model.predict_proba(
    X_test
)[:, 1]

logistic_auc = roc_auc_score(
    y_test,
    logistic_probability
)

print(
    f"Logistic Regression ROC-AUC: {logistic_auc:.4f}"
)


# ---------------------------------------------------------
# 8. RANDOM FOREST
# ---------------------------------------------------------

print("\nTraining Random Forest...")

random_forest_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced"
            )
        )
    ]
)

random_forest_model.fit(X_train, y_train)

rf_probability = random_forest_model.predict_proba(
    X_test
)[:, 1]

rf_auc = roc_auc_score(
    y_test,
    rf_probability
)

print(
    f"Random Forest ROC-AUC: {rf_auc:.4f}"
)


# ---------------------------------------------------------
# 9. XGBOOST
# ---------------------------------------------------------

print("\nTraining XGBoost...")

xgb_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss",
                n_jobs=-1
            )
        )
    ]
)

xgb_model.fit(X_train, y_train)

xgb_probability = xgb_model.predict_proba(
    X_test
)[:, 1]

xgb_auc = roc_auc_score(
    y_test,
    xgb_probability
)

print(
    f"XGBoost ROC-AUC: {xgb_auc:.4f}"
)


# ---------------------------------------------------------
# 10. MODEL COMPARISON
# ---------------------------------------------------------

results = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],
        "ROC-AUC": [
            logistic_auc,
            rf_auc,
            xgb_auc
        ]
    }
)

results = results.sort_values(
    "ROC-AUC",
    ascending=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(results.to_string(index=False))


# ---------------------------------------------------------
# 11. SELECT BEST MODEL
# ---------------------------------------------------------

best_model_name = results.iloc[0]["Model"]

if best_model_name == "Logistic Regression":
    best_model = logistic_model

elif best_model_name == "Random Forest":
    best_model = random_forest_model

else:
    best_model = xgb_model


print(
    f"\nBest Model: {best_model_name}"
)


# ---------------------------------------------------------
# 12. SAVE MODEL
# ---------------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

model_bundle = {
    "model": best_model,
    "features": features,
    "target": TARGET,
    "model_name": best_model_name
}

joblib.dump(
    model_bundle,
    MODEL_PATH
)

print(
    f"\nModel saved successfully:"
)

print(MODEL_PATH)

print("\nTraining completed!")


