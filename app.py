# app.py
from flask import Flask, request, jsonify
from joblib import load
import logging
import pandas as pd

app = Flask(__name__)

# Load the model
model = load('churn_model.joblib')

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Root route
@app.route('/')
def home():
    return "Welcome to the Customer Churn Prediction API! Use the /predict endpoint to make predictions."

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = pd.DataFrame([data])
        input_data = pd.get_dummies(input_data, drop_first=True)  # Ensure same preprocessing as training
       
        # Ensure all feature names are present
        feature_names = model.feature_names_in_
        for feature in feature_names:
            if feature not in input_data.columns:
                input_data[feature] = 0

        input_data = input_data[feature_names]

        prediction = model.predict(input_data)[0]
        return jsonify({'churn': bool(prediction)})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)