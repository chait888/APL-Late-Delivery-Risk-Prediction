
import sys
import os

# Add src folder to Python path
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src"
    )
)

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from feature_engineering import engineer_features


# ====================================
# SETTINGS
# ====================================

DATA_PATH = "data/APL_Logistics.csv"
MODEL_PATH = "models/late_delivery_model.pkl"

TARGET = "Late_delivery_risk"


print("====================================")
print("MODEL EVALUATION STARTED")
print("====================================")


# ====================================
# 1. LOAD DATA
# ====================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH,
    encoding="latin1"
)

print("Dataset shape:", df.shape)


# ====================================
# 2. FEATURE ENGINEERING
# ====================================

print("\nCreating features...")

df = engineer_features(df)


# ====================================
# 3. LOAD TRAINED MODEL
# ====================================

print("\nLoading trained model...")

bundle = joblib.load(
    MODEL_PATH
)

model = bundle["model"]

features = bundle["features"]

model_name = bundle["model_name"]

target = bundle["target"]


print("Best model:", model_name)


# ====================================
# 4. PREPARE DATA
# ====================================

X = df[features]

y = df[target]


print("\nNumber of records:", len(X))

print("Number of features:", len(features))


# ====================================
# 5. CREATE SAME TEST SPLIT
# ====================================

print("\nCreating test dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Training rows:", len(X_train))

print("Testing rows:", len(X_test))


# ====================================
# 6. MAKE TEST PREDICTIONS
# ====================================

print("\nMaking predictions on test data...")

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)[:, 1]


# ====================================
# 7. MODEL PERFORMANCE
# ====================================

print("\n====================================")

print("MODEL PERFORMANCE")

print("====================================")


roc_auc = roc_auc_score(
    y_test,
    y_probability
)


precision = precision_score(
    y_test,
    y_pred
)


recall = recall_score(
    y_test,
    y_pred
)


f1 = f1_score(
    y_test,
    y_pred
)


print(
    "\nROC-AUC :",
    round(roc_auc, 4)
)


print(
    "Precision:",
    round(precision, 4)
)


print(
    "Recall   :",
    round(recall, 4)
)


print(
    "F1 Score :",
    round(f1, 4)
)


# ====================================
# 8. CLASSIFICATION REPORT
# ====================================

print("\n====================================")

print("CLASSIFICATION REPORT")

print("====================================")


print(
    classification_report(
        y_test,
        y_pred
    )
)


# ====================================
# 9. CONFUSION MATRIX
# ====================================

print("\nCreating confusion matrix...")


cm = confusion_matrix(
    y_test,
    y_pred
)


display = ConfusionMatrixDisplay(
    confusion_matrix=cm
)


display.plot()


plt.title(
    "Confusion Matrix - " + model_name
)


plt.tight_layout()


plt.savefig(
    "reports/Confusion_Matrix.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nConfusion matrix saved successfully."
)


print(
    "Location: reports/Confusion_Matrix.png"
)


# ====================================
# 10. FINAL MESSAGE
# ====================================

print("\n====================================")

print("MODEL EVALUATION COMPLETED")

print("====================================")

