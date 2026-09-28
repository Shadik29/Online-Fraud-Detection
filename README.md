# 🛡️ Online Payments Fraud Detection

A Machine Learning web application that detects fraudulent online payment transactions in real-time using a **Random Forest Classifier** trained on 6.3 million real transaction records.

**Model Accuracy: 99.24%**

-----

## 📁 Project Structure

```
online payments fraud detection/
│
├── data/
│   └── PS_20174392719_1491204439457_log.csv
│
├── flask/
│   ├── templates/
│   │   ├── home.html
│   │   ├── predict.html
│   │   └── submit.html
│   ├── app.py
│   ├── app_ibm.py
│   └── payments.pkl
│
├── training/
│   ├── ONLINE PAYMENTS FRAUD DETECTION.ipynb
│   └── payments.pkl
│
└── training_ibm/
    └── online payments fraud prediction using ibm.ipynb
```

-----

## 🔧 Tech Stack

- **Python** - Programming Language
- **Scikit-learn** - Machine Learning
- **Random Forest** - ML Algorithm
- **Flask** - Web Framework
- **HTML/CSS** - Frontend
- **Pandas/Numpy** - Data Processing

-----

## 📊 Model Performance

|Metric           |Score |
|-----------------|------|
|Accuracy         |99.24%|
|Precision (Fraud)|98%   |
|Recall (Fraud)   |99%   |
|F1-Score         |98%   |

-----

## 🚀 How to Run

### Step 1 — Install Required Packages

```bash
pip3 install -r requirements.txt
```

### Step 2 — Train the Model

```bash
cd training
python3 train_model.py
```

### Step 3 — Run the Web App

```bash
cd ../flask
python3 app.py
```

### Step 4 — Open in Browser

```
http://127.0.0.1:5000
```

-----

## 🧪 Test Cases

### ✅ Legitimate Transaction

|Field               |Value    |
|--------------------|---------|
|Step                |1        |
|Type                |PAYMENT  |
|Amount              |9839.64  |
|Sender Old Balance  |170136.00|
|Sender New Balance  |160296.36|
|Receiver Old Balance|0.00     |
|Receiver New Balance|0.00     |

### 🚨 Fraudulent Transaction

|Field               |Value   |
|--------------------|--------|
|Step                |1       |
|Type                |TRANSFER|
|Amount              |181.00  |
|Sender Old Balance  |181.00  |
|Sender New Balance  |0.00    |
|Receiver Old Balance|0.00    |
|Receiver New Balance|0.00    |

-----

👨‍💻 My Contribution
Data preprocessing and cleaning
Feature selection
Model building and evaluation

## 📌 GitHub Repository

```
https://github.com/Sarvepallisai/ONLINE_PAYMENTS_FRAUD_DETECTION
```
