# Machine Learning-Based Late Delivery Risk Prediction in Global Supply Chain Operations

## 📌 Project Overview

This project develops a Machine Learning-based solution to predict the risk of late deliveries in global supply chain operations.

The objective is to identify orders that are likely to experience delivery delays so that logistics teams can take proactive actions such as shipment monitoring, route review, carrier escalation, and customer communication.

The project combines **Business Analytics, Supply Chain Analytics, Data Analysis, Machine Learning, and Business Intelligence** to support data-driven logistics decision-making.

---

## 🎯 Business Problem

Late deliveries can result in:

* Customer dissatisfaction
* SLA violations
* Financial penalties
* Increased logistics costs
* Customer churn
* Poor supply chain performance

Traditional reporting mainly analyzes delays after they occur.

This project focuses on **predictive analytics**, allowing organizations to identify potential delivery risks before they become major operational problems.

---

## 🎯 Project Objectives

1. Analyze historical logistics and delivery data.
2. Identify factors associated with late delivery risk.
3. Engineer meaningful business features.
4. Train multiple Machine Learning models.
5. Compare model performance using appropriate evaluation metrics.
6. Predict late delivery probability.
7. Classify shipments into Low, Medium, and High risk.
8. Explain important factors influencing predictions using SHAP.
9. Develop an interactive Streamlit dashboard.
10. Provide actionable business recommendations for logistics management.

---

## 📊 Dataset

The dataset contains approximately **180,000+ logistics order records** with information related to:

* Customer information
* Product information
* Order information
* Shipping mode
* Shipping duration
* Sales
* Discounts
* Profit
* Market
* Region
* Customer segment
* Delivery risk

### Target Variable

`Late_delivery_risk`

Where:

* `0` = No late delivery risk
* `1` = Late delivery risk

---

## 🔍 Exploratory Data Analysis

The project analyzes relationships between late delivery risk and:

* Shipping Mode
* Order Region
* Customer Segment
* Scheduled Shipping Days
* Order Quantity
* Discount Rate
* Sales
* Profit

EDA visualizations are generated to identify operational patterns and potential risk factors.

---

## ⚙️ Feature Engineering

Additional business-oriented features were created:

### Shipping Pressure

Measures order quantity relative to scheduled shipping time.

### High Quantity Flag

Identifies orders with relatively high quantities.

### High Discount Flag

Identifies orders with higher discount rates.

### Express Shipping Flag

Identifies express shipping orders.

### Order Complexity

Combines order quantity and discount information.

### Profit Margin

Measures profit relative to sales.

---

## 🤖 Machine Learning Models

Three Machine Learning algorithms are considered:

### 1. Logistic Regression

Used as a baseline classification model.

### 2. Random Forest

Used to capture nonlinear relationships and interactions between variables.

### 3. XGBoost

Used as an advanced gradient boosting model for improved predictive performance.

---

## 📈 Model Evaluation

The models are evaluated using:

* ROC-AUC
* Precision
* Recall
* F1 Score
* Confusion Matrix

The model with the strongest ROC-AUC performance is selected as the final model.

---

## 🧠 Explainable AI

SHAP (SHapley Additive exPlanations) is used to understand which features have the greatest influence on late delivery predictions.

This improves model transparency and helps logistics managers understand the reasons behind predicted delivery risk.

Generated visualization:

`reports/SHAP_Summary.png`

---

## 🚚 Streamlit Dashboard

An interactive Streamlit dashboard allows users to enter order information and receive:

* Late delivery probability
* Risk classification
* Model prediction
* Business recommendations

### Risk Classification

| Probability | Risk Level |
| ----------- | ---------- |
| Below 33%   | Low        |
| 33%–66%     | Medium     |
| Above 66%   | High       |

---

## 💼 Business Recommendations

### Low Risk

Continue normal shipment monitoring.

### Medium Risk

Monitor the shipment closely and consider proactive customer communication.

### High Risk

Consider:

* Priority shipment handling
* Route review
* Carrier escalation
* Proactive customer communication
* Additional operational monitoring

---

## 🛠️ Technology Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost

### Explainable AI

* SHAP

### Dashboard

* Streamlit

### Model Deployment

* Joblib

---

## 📁 Project Structure

```text
APL_Late_Delivery_Project/
│
├── data/
│   └── APL_Logistics.csv
│
├── notebooks/
│   ├── 01_EDA.py
│   ├── 02_Feature_Engineering.py
│   ├── 03_Model_Training.py
│   └── 04_Model_Evaluation.py
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── explain_model.py
│
├── models/
│   └── late_delivery_model.pkl
│
├── reports/
│   ├── SHAP_Summary.png
│   └── Confusion_Matrix.png
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ How to Run the Project

### Step 1: Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python src/train_model.py
```

### Step 3: Run SHAP Analysis

```bash
python src/explain_model.py
```

### Step 4: Run Model Evaluation

```bash
python notebooks/04_Model_Evaluation.py
```

### Step 5: Launch the Dashboard

```bash
python -m streamlit run app.py
```

---

## 📌 Key Business Value

This project demonstrates how predictive analytics can support supply chain decision-making by moving from:

**Descriptive Analytics → Diagnostic Analytics → Predictive Analytics → Prescriptive Business Action**

Instead of only reporting which shipments were late, the solution attempts to identify **which shipments are at risk of becoming late** and provides recommendations for proactive intervention.

---

## 🎓 Skills Demonstrated

* Business Analytics
* Supply Chain Analytics
* Predictive Analytics
* Data Cleaning
* Exploratory Data Analysis
* Feature Engineering
* Machine Learning
* Model Evaluation
* Explainable AI
* Risk Classification
* Data Visualization
* Dashboard Development
* Business Decision Making
* Python
* SQL-oriented analytical thinking

---

## 👨‍💼 Business Analytics Perspective

The project demonstrates the application of analytics to a real-world supply chain problem.

It connects technical Machine Learning outputs with business decisions by translating predicted delivery risk into operational actions.

The solution can help logistics teams prioritize high-risk shipments and improve proactive supply chain management.

---

## 🚀 Future Enhancements

Potential future improvements include:

* Real-time shipment data integration
* Route-level risk prediction
* Carrier performance analysis
* Time-series forecasting
* Automated email alerts for high-risk shipments
* Cloud deployment
* Real-time logistics monitoring
* Advanced optimization for shipment prioritization

---

## 📌 Project Type

**MBA Business Analytics | Supply Chain Analytics | Machine Learning | Predictive Analytics**
