import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# 1. Load Dataset
# -----------------------------

DATA_PATH = "data/APL_Logistics.csv"

df = pd.read_csv(
    DATA_PATH,
    encoding="latin1"
)

print("=" * 60)
print("APL LOGISTICS - EXPLORATORY DATA ANALYSIS")
print("=" * 60)


# -----------------------------
# 2. Basic Information
# -----------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)


# -----------------------------
# 3. Missing Values
# -----------------------------

print("\nMissing Values:")
missing = df.isnull().sum()

print(
    missing[missing > 0].sort_values(
        ascending=False
    )
)


# -----------------------------
# 4. Duplicate Records
# -----------------------------

print("\nDuplicate Rows:")
print(df.duplicated().sum())


# -----------------------------
# 5. Target Distribution
# -----------------------------

print("\nLate Delivery Risk Distribution:")
print(
    df["Late_delivery_risk"].value_counts()
)

print("\nLate Delivery Risk Percentage:")
print(
    df["Late_delivery_risk"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)


# -----------------------------
# 6. Target Visualization
# -----------------------------

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Late_delivery_risk"
)

plt.title("Late Delivery Risk Distribution")
plt.xlabel("Late Delivery Risk")
plt.ylabel("Number of Orders")

plt.tight_layout()
plt.show()


# -----------------------------
# 7. Shipping Mode Analysis
# -----------------------------

shipping_risk = pd.crosstab(
    df["Shipping Mode"],
    df["Late_delivery_risk"],
    normalize="index"
) * 100

print("\nLate Delivery Risk by Shipping Mode:")
print(shipping_risk.round(2))


shipping_risk.plot(
    kind="bar",
    figsize=(9, 6)
)

plt.title("Late Delivery Risk by Shipping Mode")
plt.xlabel("Shipping Mode")
plt.ylabel("Percentage")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# -----------------------------
# 8. Region Analysis
# -----------------------------

region_risk = pd.crosstab(
    df["Order Region"],
    df["Late_delivery_risk"],
    normalize="index"
) * 100

print("\nLate Delivery Risk by Region:")
print(region_risk.round(2))


region_risk.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Late Delivery Risk by Order Region")
plt.xlabel("Order Region")
plt.ylabel("Percentage")
plt.xticks(rotation=60)

plt.tight_layout()
plt.show()


# -----------------------------
# 9. Customer Segment Analysis
# -----------------------------

segment_risk = pd.crosstab(
    df["Customer Segment"],
    df["Late_delivery_risk"],
    normalize="index"
) * 100

print("\nLate Delivery Risk by Customer Segment:")
print(segment_risk.round(2))


segment_risk.plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Late Delivery Risk by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Percentage")

plt.tight_layout()
plt.show()


# -----------------------------
# 10. Scheduled Shipping Days
# -----------------------------

scheduled_risk = pd.crosstab(
    df["Days for shipment (scheduled)"],
    df["Late_delivery_risk"],
    normalize="index"
) * 100

print("\nLate Delivery Risk by Scheduled Shipping Days:")
print(scheduled_risk.round(2))


scheduled_risk.plot(
    kind="bar",
    figsize=(9, 6)
)

plt.title(
    "Late Delivery Risk by Scheduled Shipping Days"
)

plt.xlabel("Scheduled Shipping Days")
plt.ylabel("Percentage")

plt.tight_layout()
plt.show()


# -----------------------------
# 11. Order Quantity Analysis
# -----------------------------

quantity_risk = (
    df.groupby("Late_delivery_risk")["Order Item Quantity"]
    .mean()
)

print("\nAverage Order Quantity by Risk:")
print(quantity_risk)


# -----------------------------
# 12. Discount Analysis
# -----------------------------

discount_risk = (
    df.groupby("Late_delivery_risk")
    ["Order Item Discount Rate"]
    .mean()
)

print("\nAverage Discount Rate by Risk:")
print(discount_risk)


# -----------------------------
# 13. Sales Analysis
# -----------------------------

sales_risk = (
    df.groupby("Late_delivery_risk")["Sales"]
    .mean()
)

print("\nAverage Sales by Risk:")
print(sales_risk)


# -----------------------------
# 14. Final Summary
# -----------------------------

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nBusiness Questions Analyzed:")
print("1. What percentage of orders are at late-delivery risk?")
print("2. Which shipping modes have higher risk?")
print("3. Which regions have higher risk?")
print("4. Does customer segment affect delivery risk?")
print("5. Does scheduled shipping time affect risk?")
print("6. Does order quantity relate to delivery risk?")
print("7. Does discount rate relate to delivery risk?")
print("8. Does sales value differ by delivery risk?")


