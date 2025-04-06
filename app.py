from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get values from form
        name = request.form['name']
        step = float(request.form['step'])
        trans_type = request.form['type']
        amount = float(request.form['amount'])
        oldbalanceOrg = float(request.form['oldbalanceOrg'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])

        # One-hot encode type manually
        type_list = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
        encoded_type = [1 if trans_type == t else 0 for t in type_list]

        # Construct feature vector
        features = [step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest] + encoded_type
        features = np.array(features).reshape(1, -1)

        # Scale and predict
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        result = "Fraudulent Transaction" if prediction == 1 else "Not Fraudulent"

        return render_template('predict.html',
                               prediction_text=result,
                               name=name,
                               step=step,
                               trans_type=trans_type,
                               amount=amount,
                               oldbalanceOrg=oldbalanceOrg,
                               newbalanceOrig=newbalanceOrig,
                               oldbalanceDest=oldbalanceDest,
                               newbalanceDest=newbalanceDest)
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
