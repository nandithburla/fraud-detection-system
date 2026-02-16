# 💳 Fraud Detection System (Machine Learning + Streamlit)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/Model-RandomForest-orange)
![Imbalance](https://img.shields.io/badge/Handling-SMOTE-red)
![Frontend](https://img.shields.io/badge/UI-Streamlit-green)

An **end-to-end fraud detection system** built using real-world credit card transaction data.  
The system handles extreme class imbalance (0.16% fraud cases), compares multiple models, and deploys the final model using **Streamlit**.

---

## 🚀 Features

- 🔎 Exploratory Data Analysis (EDA)
- ⚖️ Extreme class imbalance handling using **SMOTE**
- 🤖 Model comparison:
  - Logistic Regression
  - Random Forest
  - XGBoost
- 🎯 Threshold tuning for better precision-recall tradeoff
- 💰 Financial impact analysis
- 🌐 Deployed interactive web app (Streamlit)
- 📦 Model serialization using Joblib

---

## 🧠 How the System Works

### 1️⃣ Data Processing
- Removed duplicates
- Scaled `Amount` and `Time` separately using StandardScaler
- PCA-based features (V1–V28) used directly
- Stratified train-test split

### 2️⃣ Imbalance Handling
- Fraud ratio ≈ **0.16%**
- Applied **SMOTE only on training data**

### 3️⃣ Model Training
Compared:
- Logistic Regression
- Random Forest
- XGBoost

### 4️⃣ Final Model Selection

**Random Forest was selected due to:**
- Very high precision (0.92)
- Extremely low false positives (6 only)
- Strong recall (0.77)
- Best real-world deployment balance

---

## 📊 Final Model Performance (Random Forest)

- **Precision (Fraud):** 0.92  
- **Recall (Fraud):** 0.77  
- **False Positives:** 6  
- **ROC-AUC:** ~0.88  

### 💰 Financial Impact

- ~$8,900 fraud prevented (test set)
- Minimal customer inconvenience
- Production-ready balance between detection and usability

---

## 🏗️ Project Structure

```text
fraud-detection-system/
│
├── app.py                     # Streamlit web application
├── predict.py                 # Standalone prediction script
├── fraud_detection_model.pkl  # Trained Random Forest model
├── amount_scaler.pkl          # Scaler for transaction amount
├── time_scaler.pkl            # Scaler for transaction time
├── Fraud_Detection_System.ipynb
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run Locally

### 1️⃣ Clone Repository
```bash
git clone https://github.com/<your-username>/fraud-detection-system.git
cd fraud-detection-system
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run Streamlit App
```bash
streamlit run app.py
```

Then open:

👉 http://localhost:8501

---

## 🔍 Prediction Logic

The model expects:
```
[Amount, Time, V1, V2, ..., V28]
```

- `Amount` and `Time` are scaled
- PCA components used as provided
- Final input passed to Random Forest classifier

---

## 📌 Business Perspective

This system prioritizes:

- High precision to reduce false fraud alerts
- Minimal customer disruption
- Operational efficiency
- Practical deployability

Unlike academic models focused only on recall, this system balances fraud detection with real-world constraints.

---

## 🚧 Future Improvements

- SHAP explainability
- FastAPI REST API version
- Cloud deployment
- Real-time fraud scoring
- Model monitoring pipeline

---

## 👤 Author

**Nandu**  
GitHub: https://github.com/nandithburla
