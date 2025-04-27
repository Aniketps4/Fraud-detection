# Fraud-detection


---

# 📊 Fraud Detection using Machine Learning

This project focuses on building a machine learning model for **proactive fraud detection** using transaction data.  
It involves data cleaning, model development, evaluation, and insight generation to help financial companies combat fraudulent activities.

---

## 📁 Project Structure

- `fraud_detection.ipynb` — Main Jupyter Notebook with all code
- `Fraud.csv` — Dataset 

---

## 🚀 Getting Started

### Install Requirements

Run the following to install necessary libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost
```

Or, if using Google Colab, use:

```python
!pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost
```

---

## 📚 Dataset Description

- **Size:** 6,362,620 rows × 10 columns
- **Problem Type:** Binary Classification
- **Target Variable:** `isFraud`
  - 1 = Fraudulent Transaction
  - 0 = Legitimate Transaction

---

## 🛠️ Main Steps

1. **Data Cleaning**
   - Handling missing values
   - Treating outliers
   - Checking and reducing multicollinearity using VIF

2. **Feature Engineering**
   - One-Hot Encoding of categorical variables
   - Feature scaling using StandardScaler

3. **Model Building**
   - Random Forest Classifier
   - XGBoost Classifier (with optional GPU acceleration)

4. **Model Evaluation**
   - Confusion Matrix
   - Classification Report (Precision, Recall, F1-Score)
   - ROC-AUC Curve

5. **Insights**
   - Feature importance analysis
   - Identification of key fraud indicators
   - Actionable recommendations for the business

---

## 📈 Model Performance

- **ROC-AUC Score:** Achieved (fill in your value from your output)  
- **Key Metrics:** High Recall and Precision achieved to effectively detect fraud.

---

## 🔍 Key Findings

- Features such as **Transaction Amount**, **Old Balance Difference**, and **Transaction Type** were critical indicators of fraud.
- Larger transactions and suspicious balance changes were highly correlated with fraudulent activities.

---

## 🛡️ Recommendations

- Implement real-time fraud scoring and alerts
- Enable multi-factor authentication for risky transactions
- Create customer behavioral profiles for anomaly detection
- Regular monitoring and retraining of fraud detection models

---

## 📊 Future Work

- Deploy the model via Flask API or Streamlit dashboard
- Handle full dataset without sampling using cloud-based GPU
- Perform hyperparameter tuning (Grid Search / Random Search)
- Explore anomaly detection models (e.g., Isolation Forests)

---

## 🤝 Acknowledgements

This project was developed as part of a data science case study for proactive fraud detection.

---

## 📬 Contact

Feel free to connect:

- **Name:** Aniket Pasi
- **Email:** aniket.ps.1998@gmail.com

---

# ✅ Quick Commands

If you clone this repo:

```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
jupyter notebook

---

# ⚡ Done!
