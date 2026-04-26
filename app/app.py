import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app.features import engineer_features, MODEL_FEATURES
from app.inference import predict_churn

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Risk Dashboard",
    page_icon="📡",
    layout="wide"
)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    path = os.path.join(BASE_DIR, 'data', 'processed', 'telco_churn_processed.csv')
    df = pd.read_csv(path)
    df = engineer_features(df)
    return df

df = load_data()

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("Telco Churn Risk")
page = st.sidebar.radio("Navigate", [
    " Overview & Insights",
    " Single Customer Predict",
    " Revenue at Risk",
    " Model Info"
])

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview & Insights
# ═════════════════════════════════════════════════════════════════════════════
if page == " Overview & Insights":
    st.title(" Telco Churn Risk Dashboard")
    st.markdown("*End-to-end churn analysis combining SQL-like aggregations with ML models*")
    st.divider()

    # KPI cards
    total = len(df)
    churned = df['Churn'].sum()
    churn_rate = df['Churn'].mean() * 100
    mrr = df['MonthlyCharges'].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churned Customers", f"{int(churned):,}")
    c3.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
    c4.metric("Total MRR", f"${mrr:,.0f}")

    st.divider()

    # Business insight box
    st.warning("""
    💡 **Key Business Insight**
    If month-to-month fiber customers churn just **5% more**, that's an additional
    **$9,224/month → $110,690/year** in lost revenue.
    The highest-risk segment (Month-to-Month + Fiber + 0–12 months) has a **70.2% churn rate**.
    """)

    st.divider()

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Rate by Contract Type")
        contract_churn = df.groupby('Contract')['Churn'].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(contract_churn.index, contract_churn.values,
                      color=['#2ecc71', '#f39c12', '#e74c3c'])
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("Churn Rate by Contract")
        for bar, val in zip(bars, contract_churn.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Churn Rate by Tenure Bucket")
        tenure_churn = df.groupby('tenure_bucket', observed=True)['Churn'].mean() * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(tenure_churn.index, tenure_churn.values,
                      color=['#e74c3c', '#f39c12', '#2ecc71'])
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("Churn Rate by Tenure")
        for bar, val in zip(bars, tenure_churn.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
        st.pyplot(fig)
        plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Single Customer Predict
# ═════════════════════════════════════════════════════════════════════════════
elif page == " Single Customer Predict":
    st.title(" Single Customer Churn Prediction")
    st.markdown("Enter customer details to get an instant churn risk score.")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)

    with col2:
        total_charges = st.number_input("Total Charges ($)", 0.0, 9000.0,
                                         float(monthly_charges * tenure))
        senior = st.selectbox("Senior Citizen", [0, 1])

    with col3:
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        contract = st.selectbox("Contract Type",
                                ["Month-to-month", "One year", "Two year"])

    st.divider()

    if st.button("🔮 Predict Churn Risk", use_container_width=True):
        customer = {
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'is_fiber': 1 if internet == "Fiber optic" else 0,
            'is_month_to_month': 1 if contract == "Month-to-month" else 0,
            'SeniorCitizen': senior
        }

        prob, risk = predict_churn(customer)

        st.subheader(f"Result: {risk}")
        st.progress(int(prob))
        st.metric("Churn Probability", f"{prob}%")

        if prob >= 70:
            st.error(" Immediate action recommended — offer retention discount or contract upgrade.")
        elif prob >= 40:
            st.warning(" Monitor this customer — consider a loyalty reward.")
        else:
            st.success(" Low risk — customer likely to stay.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Revenue at Risk
# ═════════════════════════════════════════════════════════════════════════════
elif page == " Revenue at Risk":
    st.title("Revenue at Risk by Segment")
    st.markdown("*SQL-style aggregation: Contract × Internet Service × Tenure Bucket*")
    st.divider()

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        contract_filter = st.multiselect("Filter by Contract",
                                          df['Contract'].unique().tolist(),
                                          default=df['Contract'].unique().tolist())
    with col2:
        internet_filter = st.multiselect("Filter by Internet Service",
                                          df['InternetService'].unique().tolist(),
                                          default=df['InternetService'].unique().tolist())

    filtered = df[
        df['Contract'].isin(contract_filter) &
        df['InternetService'].isin(internet_filter)
    ]

    revenue_risk = filtered.groupby(
        ['Contract', 'InternetService', 'tenure_bucket'], observed=True
    ).agg(
        Total_Customers=('customerID', 'count'),
        Churned=('Churn', 'sum'),
        Churn_Rate=('Churn', 'mean'),
        Avg_Monthly_Charges=('MonthlyCharges', 'mean'),
    ).reset_index()

    revenue_risk['Monthly_Revenue_at_Risk'] = (
        revenue_risk['Churned'] * revenue_risk['Avg_Monthly_Charges']
    ).round(2)
    revenue_risk['Churn_Rate'] = (revenue_risk['Churn_Rate'] * 100).round(2)
    revenue_risk = revenue_risk.sort_values('Monthly_Revenue_at_Risk', ascending=False)

    st.dataframe(revenue_risk, use_container_width=True)

    total_risk = revenue_risk['Monthly_Revenue_at_Risk'].sum()
    st.metric("Total Monthly Revenue at Risk (filtered)", f"${total_risk:,.2f}")

    csv = revenue_risk.to_csv(index=False)
    st.download_button(" Download CSV", csv, "revenue_at_risk.csv", "text/csv")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Model Info
# ═════════════════════════════════════════════════════════════════════════════
elif page == " Model Info":
    st.title(" Model Information")
    st.divider()

    st.subheader("Model Comparison")
    metrics = pd.DataFrame({
        'Model': ['Logistic Regression  (chosen)', 'Random Forest'],
        'Accuracy':  [0.7246, 0.7665],
        'Precision': [0.4888, 0.5728],
        'Recall':    [0.8182, 0.4733],
        'ROC-AUC':   [0.8305, 0.7859]
    })
    st.dataframe(metrics, use_container_width=True)

    st.info("""
    **Why Logistic Regression?**
    Recall (0.82) is the most important metric for churn — it's better to flag a
    potential churner incorrectly than to miss one. LR also gives well-calibrated
    probabilities, making the churn score more interpretable for business teams.
    """)

    st.subheader("Features Used")
    st.code(str(MODEL_FEATURES))

    st.subheader("Limitations")
    st.markdown("""
    - Trained on a single telecom dataset — may not generalise across regions
    - Does not use satisfaction scores or support ticket history
    - Class imbalance handled via `class_weight='balanced'`
    """)

    st.subheader(" Roadmap")
    st.markdown("""
    - [ ] Add PostgreSQL backend for real-time customer scoring
    - [ ] Add batch scoring API with FastAPI
    - [ ] Integrate SHAP for deeper model explainability
    - [ ] Add customer lifetime value (CLV) estimation
    """)
