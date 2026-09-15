import joblib
import pandas as pd
import streamlit as st

from src.feature_engineering import engineer_features


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

DATA_PATH = "data/APL_Logistics.csv"
MODEL_PATH = "models/late_delivery_model.pkl"


st.set_page_config(
    page_title="APL Logistics Late Delivery Prediction",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 25px;
    }

    .risk-high,
    .risk-medium,
    .risk-low {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 30px;
        font-weight: bold;
    }

    .risk-high {
        background-color: #ffebee;
        border: 2px solid #e53935;
        color: #b71c1c;
    }

    .risk-medium {
        background-color: #fff8e1;
        border: 2px solid #f9a825;
        color: #e65100;
    }

    .risk-low {
        background-color: #e8f5e9;
        border: 2px solid #43a047;
        color: #1b5e20;
    }

    .section-title {
        font-size: 26px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH, encoding="latin1")
    data.columns = data.columns.str.strip()
    return data


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    df = load_data()
    bundle = load_model()

except FileNotFoundError as error:
    st.error(f"Required file not found: {error}")
    st.stop()

except Exception as error:
    st.error(f"Error loading application files: {error}")
    st.stop()


# ------------------------------------------------------------
# Validate model bundle
# ------------------------------------------------------------

if not isinstance(bundle, dict):
    st.error(
        "The model file must contain a dictionary with "
        "'model' and 'features' keys."
    )
    st.stop()


if "model" not in bundle or "features" not in bundle:
    st.error(
        "The model bundle must contain 'model' and 'features'."
    )
    st.stop()


model = bundle["model"]
features = bundle["features"]
model_name = bundle.get("model_name", "Trained Model")


# ------------------------------------------------------------
# Validate dataset columns
# ------------------------------------------------------------

required_columns = [
    "Late_delivery_risk",
    "Shipping Mode",
    "Days for shipment (scheduled)",
    "Order Item Quantity",
    "Order Item Discount Rate",
    "Sales",
    "Order Profit Per Order",
    "Customer Segment",
    "Order Region",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

st.markdown(
    '<div class="main-title">'
    "🚚 APL Logistics Late Delivery Risk Prediction"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Machine Learning-Based Early Warning System for Global Supply Chain Operations"
    "</div>",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Historical metrics
# ------------------------------------------------------------

late_risk = pd.to_numeric(
    df["Late_delivery_risk"],
    errors="coerce",
).fillna(0)

total_orders = len(df)
late_orders = int(late_risk.sum())
on_time_orders = total_orders - late_orders
historical_late_percentage = late_risk.mean() * 100


# ------------------------------------------------------------
# KPI cards
# ------------------------------------------------------------

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        "📦 Total Orders",
        f"{total_orders:,}",
    )

with kpi2:
    st.metric(
        "⚠️ Late-Risk Orders",
        f"{late_orders:,}",
    )

with kpi3:
    st.metric(
        "📊 Historical Late Risk",
        f"{historical_late_percentage:.2f}%",
    )

with kpi4:
    st.metric(
        "🤖 Best Model",
        model_name,
    )


st.divider()


# ------------------------------------------------------------
# Sidebar inputs
# ------------------------------------------------------------

st.sidebar.title("📋 Order Information")
st.sidebar.write(
    "Enter shipment details to calculate late-delivery risk."
)


shipping_modes = sorted(
    df["Shipping Mode"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

customer_segments = sorted(
    df["Customer Segment"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

order_regions = sorted(
    df["Order Region"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)


shipping_mode = st.sidebar.selectbox(
    "🚢 Shipping Mode",
    shipping_modes,
)

scheduled_days = st.sidebar.number_input(
    "📅 Scheduled Shipping Days",
    min_value=0,
    max_value=100,
    value=2,
    step=1,
)

quantity = st.sidebar.number_input(
    "📦 Order Item Quantity",
    min_value=1,
    max_value=100,
    value=1,
    step=1,
)

discount_rate = st.sidebar.number_input(
    "💰 Order Item Discount Rate",
    min_value=0.0,
    max_value=1.0,
    value=0.10,
    step=0.01,
)

sales = st.sidebar.number_input(
    "💵 Sales",
    min_value=0.0,
    value=100.0,
    step=1.0,
)

order_profit = st.sidebar.number_input(
    "📈 Order Profit Per Order",
    value=20.0,
    step=1.0,
)

customer_segment = st.sidebar.selectbox(
    "👤 Customer Segment",
    customer_segments,
)

order_region = st.sidebar.selectbox(
    "🌎 Order Region",
    order_regions,
)

st.sidebar.divider()

predict_button = st.sidebar.button(
    "🚨 PREDICT RISK",
    use_container_width=True,
)


# ------------------------------------------------------------
# Prediction section
# ------------------------------------------------------------

st.markdown(
    '<div class="section-title">'
    "🎯 Late Delivery Risk Prediction"
    "</div>",
    unsafe_allow_html=True,
)


if predict_button:

    if df.empty:
        st.error("The dataset is empty.")
        st.stop()

    # Keep the result as a one-row DataFrame.
    # Do not change this to df.iloc[] or df.iloc[0].
    input_data = df.iloc[[0]].copy()

    input_data.loc[:, "Shipping Mode"] = shipping_mode
    input_data.loc[
        :, "Days for shipment (scheduled)"
    ] = scheduled_days
    input_data.loc[:, "Order Item Quantity"] = quantity
    input_data.loc[
        :, "Order Item Discount Rate"
    ] = discount_rate
    input_data.loc[:, "Sales"] = sales
    input_data.loc[
        :, "Order Profit Per Order"
    ] = order_profit
    input_data.loc[:, "Customer Segment"] = customer_segment
    input_data.loc[:, "Order Region"] = order_region

    try:
        input_data = engineer_features(input_data)

        missing_features = [
            feature
            for feature in features
            if feature not in input_data.columns
        ]

        if missing_features:
            st.error(
                f"Missing model features after feature engineering: "
                f"{missing_features}"
            )
            st.stop()

        X_input = input_data[features]

        probability = float(
            model.predict_proba(X_input)[0, 1]
        )

        prediction = int(
            model.predict(X_input)[0]
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.stop()

    # --------------------------------------------------------
    # Risk classification
    # --------------------------------------------------------

    if probability < 0.33:
        risk_level = "LOW"
        risk_class = "risk-low"
        recommendation = (
            "Continue normal shipment monitoring."
        )

    elif probability < 0.66:
        risk_level = "MEDIUM"
        risk_class = "risk-medium"
        recommendation = (
            "Monitor shipment closely and consider "
            "proactive customer communication."
        )

    else:
        risk_level = "HIGH"
        risk_class = "risk-high"
        recommendation = (
            "Consider priority handling, route review, "
            "carrier escalation, and proactive customer communication."
        )

    # --------------------------------------------------------
    # Risk result
    # --------------------------------------------------------

    st.markdown(
        f"""
        <div class="{risk_class}">
            🚨 {risk_level} DELIVERY RISK
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    result1, result2, result3 = st.columns(3)

    with result1:
        st.metric(
            "🎯 Late Delivery Probability",
            f"{probability:.2%}",
        )

    with result2:
        st.metric(
            "⚠️ Risk Level",
            risk_level,
        )

    with result3:
        st.metric(
            "📦 Prediction",
            "LATE" if prediction == 1 else "ON TIME",
        )

    st.write("")
    st.write("### 📊 Risk Probability")

    st.progress(
        min(max(probability, 0.0), 1.0)
    )

    st.caption(
        f"Probability of late delivery: {probability:.2%}"
    )

    # --------------------------------------------------------
    # Order assessment
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        "📋 Order Assessment"
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.write("**Shipping Mode**")
        st.write(shipping_mode)

    with col2:
        st.write("**Scheduled Days**")
        st.write(f"{scheduled_days} days")

    with col3:
        st.write("**Order Quantity**")
        st.write(quantity)

    with col4:
        st.write("**Customer Segment**")
        st.write(customer_segment)

    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.write("**Discount Rate**")
        st.write(f"{discount_rate:.0%}")

    with col6:
        st.write("**Sales**")
        st.write(f"${sales:,.2f}")

    with col7:
        st.write("**Order Profit**")
        st.write(f"${order_profit:,.2f}")

    with col8:
        st.write("**Order Region**")
        st.write(order_region)

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">'
        "💡 Recommended Action"
        "</div>",
        unsafe_allow_html=True,
    )

    if risk_level == "HIGH":
        st.error(
            "🚨 **Immediate Attention Recommended**\n\n"
            f"{recommendation}"
        )

    elif risk_level == "MEDIUM":
        st.warning(
            "⚠️ **Close Monitoring Recommended**\n\n"
            f"{recommendation}"
        )

    else:
        st.success(
            "✅ **Normal Monitoring Recommended**\n\n"
            f"{recommendation}"
        )

else:
    st.info(
        "👈 Enter the order information in the sidebar and click "
        "**PREDICT RISK** to generate a prediction."
    )


# ------------------------------------------------------------
# Historical analysis
# ------------------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-title">'
    "📈 Historical Late Delivery Analysis"
    "</div>",
    unsafe_allow_html=True,
)

chart1, chart2 = st.columns(2)


risk_summary = (
    late_risk
    .map({
        0: "On Time",
        1: "Late Risk",
    })
    .value_counts()
    .reindex(
        ["On Time", "Late Risk"],
        fill_value=0,
    )
)

with chart1:
    st.write("### 📊 Order Risk Distribution")
    st.bar_chart(risk_summary)


shipping_analysis = (
    df.groupby("Shipping Mode")["Late_delivery_risk"]
    .mean()
    .mul(100)
    .sort_values(ascending=False)
)

with chart2:
    st.write("### 🚢 Late Risk by Shipping Mode")
    st.bar_chart(shipping_analysis)


# ------------------------------------------------------------
# Model information
# ------------------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-title">'
    "🤖 Model Information"
    "</div>",
    unsafe_allow_html=True,
)

model1, model2, model3 = st.columns(3)

with model1:
    st.metric(
        "Selected Model",
        model_name,
    )

with model2:
    st.metric(
        "Model Features",
        len(features),
    )

with model3:
    st.metric(
        "Dataset Records",
        f"{len(df):,}",
    )


st.info(
    """
    **Model Performance**

    The XGBoost model achieved a **ROC-AUC of 0.7601**,
    **84.49% precision**, **55.74% recall**, and
    **67.17% F1-score** on the test dataset.

    The prediction should be used as a decision-support tool alongside
    operational and real-time logistics information.
    """
)


# ------------------------------------------------------------
# Risk classification guide
# ------------------------------------------------------------

st.markdown(
    '<div class="section-title">'
    "🚦 Risk Classification"
    "</div>",
    unsafe_allow_html=True,
)

risk1, risk2, risk3 = st.columns(3)

with risk1:
    st.success(
        """
        ### 🟢 LOW RISK

        **Probability < 33%**

        Normal shipment monitoring.
        """
    )

with risk2:
    st.warning(
        """
        ### 🟡 MEDIUM RISK

        **33% – <66%**

        Close monitoring recommended.
        """
    )

with risk3:
    st.error(
        """
        ### 🔴 HIGH RISK

        **≥66%**

        Immediate logistics attention recommended.
        """
    )


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.divider()

st.caption(
    "APL Logistics Late Delivery Prediction Dashboard | "
    "Machine Learning + Supply Chain Analytics"
)