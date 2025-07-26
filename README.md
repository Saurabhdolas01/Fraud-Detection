💼 Fraud Detection Capstone Project
This project identifies fraudulent financial transactions using supervised machine learning. It evaluates Logistic Regression, Random Forest and XGBoost models and selects XGBoost based on high recall and business cost optimization. Deployed via Streamlit with business-friendly insights.

📌 Table of Contents
- Project Overview
- Tech Stack
- Key Features
- Business Cost Optimization

🧠 Project Overview
1. Built a machine learning pipeline to detect fraud using a real-world dataset.
2. Handled imbalanced data and engineered features relevant to fraud detection.
3. Evaluated multiple models and selected XGBoost based on:
4. High recall on fraud class
5. Lower business cost from false negatives
6. Includes Streamlit dashboard and Power BI visualizations.

⚙️ Tech Stack
Tool	            Usage
Python	          Core ML modeling
Pandas, NumPy	    Data preprocessing
Scikit-learn	    Model training + evaluation
XGBoost	          Final model
Streamlit	        App deployment


🚀 Key Features
- Handles imbalanced datasets with cost-sensitive learning
- Visual fraud pattern exploration (univariate, bivariate analysis)
- Threshold tuning for financial optimization
- Interactive prediction interface with file upload support
- Final fraud counts and loss estimates


💸 Business Cost Optimization
To reflect the real-world impact of fraud, the project calculates:

- Cost per false negative (missed fraud): ₹5,000
- Cost per false positive (unnecessary investigation): ₹500
- The decision threshold is tuned to minimize total cost, not just maximize accuracy.
