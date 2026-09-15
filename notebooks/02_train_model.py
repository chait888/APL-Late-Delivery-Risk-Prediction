import sys
import os

sys.path.append(
os.path.join(
os.path.dirname(__file__),
"..",
"src"
)
)

import pandas as pd

from feature_engineering import engineer_features
from feature_engineering import get_model_features

DATA_PATH = "data/APL_Logistics.csv"

print("====================================")
print("MODEL TRAINING")
print("====================================")

print("\nLoading dataset...")

df = pd.read_csv(
DATA_PATH,
encoding="latin1"
)

print("Dataset shape:", df.shape)

print("\nApplying feature engineering...")

df = engineer_features(
df
)

features = get_model_features(
df
)

TARGET = "Late_delivery_risk"

X = df[features]

y = df[TARGET]

print("\nNumber of model features:", len(features))

print("Training data rows:", len(X))

print("\nTarget distribution:")

print(
y.value_counts()
)

print("\n====================================")
print("FEATURE PREPARATION COMPLETED")
print("====================================")

print("\nModel training is handled by:")

print("src/train_model.py")

print("\nTo train the final models, run:")

print("python src/train_model.py")

print("\n====================================")
print("NOTEBOOK MODEL TRAINING STEP COMPLETED")
print("====================================")
