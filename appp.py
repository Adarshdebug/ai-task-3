from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load('boston_model.pkl')

@app.route('/')
def home():
    return "Boston Housing Price Prediction API"

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from request
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
