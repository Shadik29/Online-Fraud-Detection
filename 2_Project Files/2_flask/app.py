"""
=============================================================
  ONLINE PAYMENTS FRAUD DETECTION - Flask Web Application
=============================================================
Run:  python app.py
Open: http://127.0.0.1:5000
"""

from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── Load trained model ──────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'payments.pkl')

with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

print("✅ Model loaded successfully!")

# ── Transaction type encoding (matches training) ────────────
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
    """Home / landing page."""
    return render_template('home.html')


@app.route('/predict', methods=['GET'])
def predict():
    """Show the prediction form."""
    return render_template('predict.html')


@app.route('/submit', methods=['POST'])
def submit():
    """Handle form submission and return fraud prediction."""
    try:
        # ── Read form values ──────────────────────────────
        step          = int(request.form['step'])
        trans_type    = request.form['type']
        amount        = float(request.form['amount'])
        old_bal_orig  = float(request.form['oldbalanceOrg'])
        new_bal_orig  = float(request.form['newbalanceOrig'])
        old_bal_dest  = float(request.form['oldbalanceDest'])
        new_bal_dest  = float(request.form['newbalanceDest'])

        # ── Encode + build feature array ──────────────────
        type_encoded = TYPE_MAPPING.get(trans_type, 3)

        features = np.array([[
            step,
            type_encoded,
            amount,
            old_bal_orig,
            new_bal_orig,
            old_bal_dest,
            new_bal_dest
        ]])

        # ── Predict ───────────────────────────────────────
        prediction  = model.predict(features)[0]
        proba       = model.predict_proba(features)[0]
        fraud_prob  = round(proba[1] * 100, 2)
        legit_prob  = round(proba[0] * 100, 2)

        if prediction == 1:
            result      = "FRAUDULENT Transaction Detected!"
            result_type = "fraud"
            icon        = "🚨"
            advice      = ("This transaction shows patterns consistent with fraud. "
                           "Do NOT proceed. Contact your bank immediately.")
        else:
            result      = "Transaction is LEGITIMATE"
            result_type = "legit"
            icon        = "✅"
            advice      = ("This transaction appears safe based on the provided details. "
                           "You may proceed normally.")

        # ── Transaction summary to show on result page ────
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
            result      = result,
            result_type = result_type,
            icon        = icon,
            advice      = advice,
            fraud_prob  = fraud_prob,
            legit_prob  = legit_prob,
            transaction = transaction
        )

    except Exception as e:
        return render_template(
            'submit.html',
            result      = f"⚠️ Error: {str(e)}",
            result_type = "error",
            icon        = "⚠️",
            advice      = "Please check your inputs and try again.",
            fraud_prob  = 0,
            legit_prob  = 0,
            transaction = {}
        )


# ── Run ─────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  Fraud Detection Web App")
    print("  Open → http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)
