"""
=============================================================
  ONLINE PAYMENTS FRAUD DETECTION - IBM Cloud Flask App
=============================================================
This file is for IBM Cloud deployment.
For local use, run app.py instead.
"""

from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── Load Model ──────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'payments.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

print("✅ Model loaded successfully!")

# ── Transaction Type Encoding ────────────────────────────────
TYPE_MAPPING = {
    'CASH_IN'  : 0,
    'CASH_OUT' : 1,
    'DEBIT'    : 2,
    'PAYMENT'  : 3,
    'TRANSFER' : 4
}

# ── Routes ──────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET'])
def predict():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        step         = int(request.form['step'])
        trans_type   = request.form['type']
        amount       = float(request.form['amount'])
        old_bal_orig = float(request.form['oldbalanceOrg'])
        new_bal_orig = float(request.form['newbalanceOrig'])
        old_bal_dest = float(request.form['oldbalanceDest'])
        new_bal_dest = float(request.form['newbalanceDest'])

        type_encoded = TYPE_MAPPING.get(trans_type, 3)

        features = np.array([[
            step, type_encoded, amount,
            old_bal_orig, new_bal_orig,
            old_bal_dest, new_bal_dest
        ]])

        prediction = model.predict(features)[0]
        proba      = model.predict_proba(features)[0]
        fraud_prob = round(proba[1] * 100, 2)
        legit_prob = round(proba[0] * 100, 2)

        if prediction == 1:
            result      = "FRAUDULENT Transaction Detected!"
            result_type = "fraud"
            icon        = "🚨"
            advice      = "This transaction shows patterns consistent with fraud. Do NOT proceed. Contact your bank immediately."
        else:
            result      = "Transaction is LEGITIMATE"
            result_type = "legit"
            icon        = "✅"
            advice      = "This transaction appears safe. You may proceed normally."

        transaction = {
            'Step'              : step,
            'Type'              : trans_type,
            'Amount'            : f"{amount:,.2f}",
            'Sender Old Balance': f"{old_bal_orig:,.2f}",
            'Sender New Balance': f"{new_bal_orig:,.2f}",
            'Recv. Old Balance' : f"{old_bal_dest:,.2f}",
            'Recv. New Balance' : f"{new_bal_dest:,.2f}",
        }

        return render_template(
            'submit.html',
            result=result, result_type=result_type,
            icon=icon, advice=advice,
            fraud_prob=fraud_prob, legit_prob=legit_prob,
            transaction=transaction
        )

    except Exception as e:
        return render_template(
            'submit.html',
            result=f"⚠️ Error: {str(e)}",
            result_type="error", icon="⚠️",
            advice="Please check your inputs and try again.",
            fraud_prob=0, legit_prob=0, transaction={}
        )

# ── IBM Cloud Entry Point ────────────────────────────────────
port = int(os.environ.get('PORT', 8080))

if __name__ == '__main__':
    print(f"\n{'='*50}")
    print(f"  IBM Cloud Fraud Detection App")
    print(f"  Running on port: {port}")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
