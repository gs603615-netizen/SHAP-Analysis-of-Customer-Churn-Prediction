Telco Customer Churn Prediction with SHAP Explainability

This project performs customer churn prediction using two classical machine learning models — Random Forest and Gradient Boosting — and enhances the analysis using SHAP explainability.

The workflow includes feature encoding, model comparison using AUC, global & local SHAP explanations, dependence plots, and automated business insights for decision-making.

🚀 Key Features

Binary classification: predict whether a customer will churn

Two ML models trained & compared:

Random Forest Classifier

Gradient Boosting Classifier

SHAP-based model explainability:

SHAP Summary Plot

SHAP Bar Plot

Local explanations (force plots) for top 5 high-risk customers

SHAP dependence plots for top 3 most important features

Automated executive summary (printed at the end)

Clean, reproducible workflow

📁 Project Structure
Telco_Churn_SHAP_Project/
│
├── src/
│   └── churn_shap.py               # Your main code
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── shap_plots/
│   ├── shap_summary.png
│   ├── shap_bar.png
│   ├── dependence_feature1.png
│   ├── dependence_feature2.png
│   ├── dependence_feature3.png
│   └── force_customer_i.png
│
├── README.md
├── requirements.txt
└── .gitignore

🧠 Data Processing

All categorical columns are automatically Label Encoded

Dataset split into:

80% training

20% testing

Target variable:

Churn

🧪 Modeling Workflow
1️⃣ Models Used
Model	Purpose
RandomForestClassifier	Strong baseline, handles non-linearity well
GradientBoostingClassifier	Boosting approach, often gives better signal
2️⃣ Model Evaluation

Metric used:

AUC (Area Under ROC Curve)

The model with higher AUC is automatically selected as the best model.

📊 SHAP Explainability
Global Explainability

SHAP Summary Plot

SHAP Bar Plot

Highlights:

Most influential features

Feature distributions

Feature directionality

Local Explainability

SHAP force plots for top 5 highest-risk churn customers

Shows:

Which features push probability toward churn

Which features reduce risk

Dependence Plots

Automatically generates dependence plots for top 3 features

Reveals feature interactions and non-linear behavior