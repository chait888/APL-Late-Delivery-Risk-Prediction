import pandas as pd


def engineer_features(df):
    """
    Create business-focused features for late delivery prediction.
    """

    df = df.copy()

    # 1. Shipping Pressure
    df["Shipping_Pressure"] = (
        df["Order Item Quantity"]
        / (df["Days for shipment (scheduled)"] + 1)
    )

    # 2. High Quantity Flag
    quantity_median = df["Order Item Quantity"].median()

    df["High_Quantity_Flag"] = (
        df["Order Item Quantity"] > quantity_median
    ).astype(int)

    # 3. High Discount Flag
    df["High_Discount_Flag"] = (
        df["Order Item Discount Rate"] > 0.20
    ).astype(int)

    # 4. Express Shipping Flag
    df["Express_Shipping_Flag"] = (
        df["Shipping Mode"]
        .astype(str)
        .str.lower()
        .str.contains("express")
    ).astype(int)

    # 5. Order Complexity
    df["Order_Complexity"] = (
        df["Order Item Quantity"]
        * (1 + df["Order Item Discount Rate"])
    )

    # 6. Profit Margin
    df["Profit_Margin"] = (
        df["Order Profit Per Order"]
        / (df["Sales"].abs() + 1)
    )

    return df


def get_model_features(df):
    """
    Return the features that should be used by the ML model.

    We intentionally exclude:
    - Late_delivery_risk → target
    - Days for shipping (real) → data leakage
    - Delivery Status → data leakage
    """

    features = [
        "Type",
        "Days for shipment (scheduled)",
        "Benefit per order",
        "Sales per customer",
        "Category Id",
        "Category Name",
        "Customer Country",
        "Customer Segment",
        "Customer State",
        "Department Id",
        "Department Name",
        "Market",
        "Order Country",
        "Order Item Discount",
        "Order Item Discount Rate",
        "Order Item Product Price",
        "Order Item Quantity",
        "Sales",
        "Order Item Total",
        "Order Profit Per Order",
        "Order Region",
        "Order State",
        "Product Price",
        "Shipping Mode",
        "Shipping_Pressure",
        "High_Quantity_Flag",
        "High_Discount_Flag",
        "Express_Shipping_Flag",
        "Order_Complexity",
        "Profit_Margin"
    ]

    return features


if __name__ == "__main__":

    DATA_PATH = "data/APL_Logistics.csv"

    print("Loading dataset...")

    df = pd.read_csv(
        DATA_PATH,
        encoding="latin1"
    )

    print("Original shape:", df.shape)

    df = engineer_features(df)

    features = get_model_features(df)

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

    print("\nFeature engineering completed successfully.")
    