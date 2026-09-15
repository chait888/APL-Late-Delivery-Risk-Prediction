import os
import joblib
import pandas as pd
import streamlit as st

from src.feature_engineering import engineer_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "APL_Logistics.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "late_delivery_model.pkl")

st.set_page_config(
    page_title="APL Logistics Late Delivery Prediction",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title { font-size: 42px; font-weight: 700; margin-bottom: 5px; }
    .subtitle { font-size: 18px; margin-bottom: 25px; }
    .risk-high, .risk-medium, .risk-low {
        padding: 25px; border-radius: 15px; text-align: center;
        font-size: 30px; font-weight: bold;
    }
    .risk-high { background-color: #ffebee; border: 2px solid #e53935; color: #b71c1c; }
    .risk-medium { background-color: #fff8e1; border: 2px solid #f9a825; color: #e65100; }
    .risk-low { background-color: #e8f5e9; border: 2px solid #43a047; color: #1b5e20; }
    .section-title { font-size: 26px; font-weight: 650; margin-top: 20px; margin-bottom: 15px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH, encoding="latin1")
    data.columns = data.columns.str.strip()
    return data


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_data()
    bundle = load_model()
except FileNotFoundError as error:
    st.error(f"Required file not found: {error}")
    st.info(
        "Ensure the repository contains `data/APL_Logistics.csv` and "
        "`models/late_delivery_model.pkl`."
    )
    st.stop()
except Exception as error:
    st.error(f"Error loading application files: {error}")
    st.stop()

if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
    st.error("The model bundle must be a dictionary containing 'model' and 'features'.")
    st.stop()

model = bundle["model"]
features = bundle["features"]
model_name = bundle.get("model_name", "Trained Model")

required_columns = [
    "Late_delivery_risk", "Shipping Mode", "Days for shipment (scheduled)",
    "Order Item Quantity", "Order Item Discount Rate", "Sales",
    "Order Profit Per Order", "Customer Segment", "Order Region",
]
missing_columns = [column for column in required_columns if column not in df.columns]
if missing_columns:
    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()

st.markdown('<div class="main-title">🚚 APL Logistics Late Delivery Risk Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Machine Learning-Based Early Warning System for Global Supply Chain Operations</div>', unsafe_allow_html=True)

late_risk = pd.to_numeric(df["Late_delivery_risk"], errors="coerce").fillna(0)
total_orders = len(df)
late_orders = int(late_risk.sum())
historical_late_percentage = late_risk.mean() * 100 if total_orders else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("📦 Total Orders", f"{total_orders:,}")
with kpi2:
    st.metric("⚠️ Late-Risk Orders", f"{late_orders:,}")
with kpi3:
    st.metric("📊 Historical Late Risk", f"{historical_late_percentage:.2f}%")
with kpi4:
    st.metric("🤖 Best Model", model_name)
st.divider()

st.sidebar.title("📋 Order Information")
st.sidebar.write("Enter shipment details to calculate late-delivery risk.")
shipping_modes = sorted(df["Shipping Mode"].dropna().astype(str).unique().tolist())
customer_segments = sorted(df["Customer Segment"].dropna().astype(str).unique().tolist())
order_regions = sorted(df["Order Region"].dropna().astype(str).unique().tolist())

shipping_mode = st.sidebar.selectbox("🚢 Shipping Mode", shipping_modes)
scheduled_days = st.sidebar.number_input("📅 Scheduled Shipping Days", 0, 100, 2, 1)
quantity = st.sidebar.number_input("📦 Order Item Quantity", 1, 100, 1, 1)
discount_rate = st.sidebar.number_input("💰 Order Item Discount Rate", 0.0, 1.0, 0.10, 0.01)
sales = st.sidebar.number_input("💵 Sales", min_value=0.0, value=100.0, step=1.0)
order_profit = st.sidebar.number_input("📈 Order Profit Per Order", value=20.0, step=1.0)
customer_segment = st.sidebar.selectbox("👤 Customer Segment", customer_segments)
order_region = st.sidebar.selectbox("🌎 Order Region", order_regions)
st.sidebar.divider()
predict_button = st.sidebar.button("🚨 PREDICT RISK", use_container_width=True)

st.markdown('<div class="section-title">🎯 Late Delivery Risk Prediction</div>', unsafe_allow_html=True)
if predict_button:
    if df.empty:
        st.error("The dataset is empty.")
        st.stop()

    input_data = df.iloc[[0]].copy()
    values = {
        "Shipping Mode": shipping_mode,
        "Days for shipment (scheduled)": scheduled_days,
        "Order Item Quantity": quantity,
        "Order Item Discount Rate": discount_rate,
        "Sales": sales,
        "Order Profit Per Order": order_profit,
        "Customer Segment": customer_segment,
        "Order Region": order_region,
    }
    for column, value in values.items():
        input_data.loc[:, column] = value

    try:
        input_data = engineer_features(input_data)
        missing_features = [feature for feature in features if feature not in input_data.columns]
        if missing_features:
            st.error(f"Missing model features after feature engineering: {missing_features}")
            st.stop()

        X_input = input_data[features]
        if not hasattr(model, "predict_proba"):
            st.error("The loaded model does not support probability predictions.")
            st.stop()
        probability = float(model.predict_proba(X_input)[0, 1])
        prediction = int(model.predict(X_input)[0])
    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.stop()

    if probability < 0.33:
        risk_level, risk_class = "LOW", "risk-low"
        recommendation = "Continue normal shipment monitoring."
    elif probability < 0.66:
        risk_level, risk_class = "MEDIUM", "risk-medium"
        recommendation = "Monitor shipment closely and consider proactive customer communication."
    else:
        risk_level, risk_class = "HIGH", "risk-high"
        recommendation = "Consider priority handling, route review, carrier escalation, and proactive customer communication."

    st.markdown(f'<div class="{risk_class}">🚨 {risk_level} DELIVERY RISK</div>', unsafe_allow_html=True)
    st.write("")
    result1, result2, result3 = st.columns(3)
    with result1:
        st.metric("🎯 Late Delivery Probability", f"{probability:.2%}")
    with result2:
        st.metric("⚠️ Risk Level", risk_level)
    with result3:
        st.metric("📦 Prediction", "LATE" if prediction == 1 else "ON TIME")

    st.write("### 📊 Risk Probability")
    st.progress(max(0.0, min(probability, 1.0)))
    st.caption(f"Probability of late delivery: {probability:.2%}")

    st.markdown('<div class="section-title">📋 Order Assessment</div>', unsafe_allow_html=True)
    assessment = [
        ("Shipping Mode", shipping_mode), ("Scheduled Days", f"{scheduled_days} days"),
        ("Order Quantity", quantity), ("Customer Segment", customer_segment),
        ("Discount Rate", f"{discount_rate:.0%}"), ("Sales", f"${sales:,.2f}"),
        ("Order Profit", f"${order_profit:,.2f}"), ("Order Region", order_region),
    ]
    for start in range(0, len(assessment), 4):
        cols = st.columns(4)
        for col, (label, value) in zip(cols, assessment[start:start + 4]):
            with col:
                st.write(f"**{label}**")
                st.write(value)

    st.markdown('<div class="section-title">💡 Recommended Action</div>', unsafe_allow_html=True)
    message = f"**{'Immediate Attention' if risk_level == 'HIGH' else 'Close Monitoring' if risk_level == 'MEDIUM' else 'Normal Monitoring'} Recommended**\n\n{recommendation}"
    if risk_level == "HIGH":
        st.error(message)
    elif risk_level == "MEDIUM":
        st.warning(message)
    else:
        st.success(message)
else:
    st.info("👈 Enter the order information in the sidebar and click **PREDICT RISK** to generate a prediction.")

st.divider()
st.markdown('<div class="section-title">📈 Historical Late Delivery Analysis</div>', unsafe_allow_html=True)
chart1, chart2 = st.columns(2)
risk_summary = late_risk.map({0: "On Time", 1: "Late Risk"}).value_counts().reindex(["On Time", "Late Risk"], fill_value=0)
with chart1:
    st.write("### 📊 Order Risk Distribution")
    st.bar_chart(risk_summary)
with chart2:
    st.write("### 🚢 Late Risk by Shipping Mode")
    shipping_analysis = df.groupby("Shipping Mode")["Late_delivery_risk"].mean().mul(100).sort_values(ascending=False)
    st.bar_chart(shipping_analysis)

st.divider()
st.markdown('<div class="section-title">🤖 Model Information</div>', unsafe_allow_html=True)
model1, model2, model3 = st.columns(3)
with model1:
    st.metric("Selected Model", model_name)
with model2:
    st.metric("Model Features", len(features))
with model3:
    st.metric("Dataset Records", f"{len(df):,}")
st.info("Model Performance\n\nThe XGBoost model achieved a **ROC-AUC of 0.7601**, **84.49% precision**, **55.74% recall**, and **67.17% F1-score** on the test dataset.\n\nThe prediction should be used as a decision-support tool alongside operational and real-time logistics information.")

st.markdown('<div class="section-title">🚦 Risk Classification</div>', unsafe_allow_html=True)
risk1, risk2, risk3 = st.columns(3)
with risk1:
    st.success("### 🟢 LOW RISK\n\n**Probability < 33%**\n\nNormal shipment monitoring.")
with risk2:
    st.warning("### 🟡 MEDIUM RISK\n\n**33% – <66%**\n\nClose monitoring recommended.")
with risk3:
    st.error("### 🔴 HIGH RISK\n\n**≥66%**\n\nImmediate logistics attention recommended.")

st.divider()
st.caption("APL Logistics Late Delivery Prediction Dashboard | Machine Learning + Supply Chain Analytics")