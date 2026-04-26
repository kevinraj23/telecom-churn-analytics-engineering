# 📡 Telco Churn Risk: SQL-Style Revenue Insights + ML Model (Python)

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=flat&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Open%20to%20Internships-brightgreen?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

> **End-to-end churn analysis combining SQL-like aggregations in pandas with ML models to estimate revenue at risk — built by an ECE + Data Science student focused on system reliability and business impact.**

---

## 🔴 Live Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-username-telecom-churn-analytics-engineering.streamlit.app)

> *https://telecom-churn-analytics-engineering-gjr4f9phdsezqhi2shw96c.streamlit.app/*

---

## 📌 Overview

This project analyses the **IBM Telco Customer Churn dataset** to answer a real business question:

> **"How much Monthly Recurring Revenue (MRR) is at risk if month-to-month fiber customers churn 5% more?"**

**Answer: $9,224/month → $110,690/year at risk.**

Using SQL-style `groupby` aggregations in pandas and two ML models (Logistic Regression + Random Forest), this project mirrors the workflow of a real data scientist at a telecom company — combining business insight with predictive modelling.

---

## ✨ Features

- 📊 **Churn probability prediction** — enter any customer's details and get instant risk score
- 💰 **Revenue-at-risk table** — SQL-style segment breakdown (Contract × Internet × Tenure)
- 🔍 **Filters** by contract type, internet service, and tenure bucket
- 📥 **Export CSV** — download the full revenue risk table
- 🧠 **Model explainability** — feature importance bar chart + confusion matrix heatmap
- 🎯 **Business insight highlight** — MRR at risk calculation for high-risk segments

---

## 🔑 Key Findings

| Segment | Churn Rate | Monthly Revenue at Risk |
|---|---|---|
| Month-to-month + Fiber + 0–12 months | **70.2%** | **$52,775** |
| Month-to-month + Fiber + 24+ months | 38.6% | $28,118 |
| Month-to-month + DSL + 0–12 months | 42.5% | $14,029 |

- Overall churn rate: **26.54%**
- New customers (0–12 months) churn at **47.68%** — nearly 1 in 2 leave early
- Fiber optic customers churn at **41.89%** vs DSL at **18.96%**

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Core language |
| pandas | Data wrangling + SQL-style aggregations |
| scikit-learn | ML models (Logistic Regression, Random Forest) |
| Streamlit | Interactive dashboard + deployment |
| joblib | Model serialization |
| matplotlib / seaborn | Visualizations |

> Designed by an **ECE + Data Science student** with focus on system reliability, lightweight deployment, and engineering-grade code structure.

---

## 📁 Project Structure

```
telecom-churn-analytics-engineering/
├── app/
│   ├── __init__.py
│   ├── app.py              # Streamlit UI (4 pages)
│   ├── features.py         # Feature engineering functions
│   └── inference.py        # Load model + predict
├── models/
│   ├── churn_model.pkl     # Trained Logistic Regression
│   └── scaler.pkl          # StandardScaler
├── notebooks/
│   └── 01_eda_telco_churn.ipynb   # Full Kaggle notebook
├── data/
│   ├── raw/                # Empty (data not tracked)
│   └── processed/
│       └── telco_churn_processed.csv
├── .streamlit/
│   └── config.toml         # Dark theme config
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Getting Started

```bash
# 1. Clone the repo
git clone https://github.com/your-username/telecom-churn-analytics-engineering.git
cd telecom-churn-analytics-engineering

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app/app.py
```

---

## 🤖 Model Performance

| Model | Accuracy | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.7246 | 0.4888 | **0.8182** | **0.8305** |
| Random Forest | 0.7665 | 0.5728 | 0.4733 | 0.7859 |

> **Logistic Regression was chosen** as the final model because **Recall (0.82)** matters most in churn — it's better to flag a potential churner incorrectly than to miss one entirely.

**Limitations:**
- Model trained on a single telecom dataset — may not generalise across regions
- Does not account for customer satisfaction scores or support ticket history
- Class imbalance handled via `class_weight='balanced'`, not SMOTE

---

## 🗺️ Roadmap

- [ ] Add PostgreSQL backend for real-time customer scoring
- [ ] Add batch scoring API with FastAPI
- [ ] Integrate SHAP for deeper model explainability
- [ ] Add customer lifetime value (CLV) estimation

---

## 🎬 Demo

> *(Add a short 10-second screen recording GIF of the main dashboard page here as `demo.gif`)*

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙋 About

Built by a final-year **ECE + Data Science** student as a portfolio project demonstrating the intersection of engineering thinking and data science.  
Open to **internship and collaboration** opportunities.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/your-profile)
