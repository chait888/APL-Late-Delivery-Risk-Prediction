import sys
import os

# Add the src directory to Python's import path
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "src"
    )
)

import pandas as pd

from feature_engineering import engineer_features, get_model_features


DATA_PATH = os.path.join("data", "APL_Logistics.csv")

print("====================================")
print("FEATURE ENGINEERING")
print("====================================")

print("\nLoading dataset...")

try:
    df = pd.read_csv(
        DATA_PATH,
        encoding="latin1"
    )

    print("Original dataset shape:", df.shape)

    print("\nApplying feature engineering...")

    df_engineered = engineer_features(df)
    features = get_model_features(df_engineered)

    print("\nNew features created:")

    new_features = [
        "Shipping_Pressure",
        "High_Quantity_Flag",
        "High_Discount_Flag",
        "Express_Shipping_Flag",
        "Order_Complexity",
        "Profit_Margin"
    ]

    for feature in new_features:
        print(" -", feature)

    print("\nTotal model features:", len(features))

    print("\nFeature engineering sample:")

    print(df_engineered[new_features].head())

    print("\n====================================")
    print("FEATURE ENGINEERING COMPLETED")
    print("====================================")

except FileNotFoundError:
    print(f"Error: Dataset not found at '{DATA_PATH}'")

except KeyError as error:
    print(f"Error: Required column or feature is missing: {error}")

except Exception as error:
    print(f"An unexpected error occurred: {error}")
