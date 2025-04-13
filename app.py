from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Allow requests from React

# Load model and scaler
model = joblib.load("C:/Users/Arshiya A/eye_detection_model.pkl")
scaler = joblib.load("C:/Users/Arshiya A/scaler.pkl")

@app.route('/predict', methods=['POST'])  # ✅ This route must match exactly
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    return jsonify({'prediction': str(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
