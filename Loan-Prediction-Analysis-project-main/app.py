from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd  # ✅ Added for DataFrame input to model

# Load the trained model
with open('random_forest_model_new.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "Hello from Flask"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    Gender = int(data['Gender'])
    Married = int(data['Married'])
    Credit_History = int(data['Credit_History'])
    Total_Income_Log = float(data['Total_Income_Log'])

    # ✅ Use DataFrame with feature names instead of raw NumPy array
    input_features = pd.DataFrame([{
        'Gender': Gender,
        'Married': Married,
        'Credit_History': Credit_History,
        'Total_Income_Log': Total_Income_Log
    }])

    prediction = model.predict(input_features)[0]
    result = "Loan Approved" if prediction == 1 else "Loan Rejected"

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
