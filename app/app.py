import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask application
app = Flask(__name__)

# Define the path to the trained model
# It's important to adjust this path based on where the app.py runs
# relative to the model file in the Docker container.
# In our Dockerfile, we'll copy the 'model' directory to the root of the app.
MODEL_PATH = 'model/wine_quality_model.pkl'

# Load the trained model
# This will be loaded once when the Flask app starts, improving performance.
try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Ensure the model is trained and saved.")
    model = None # Set model to None to handle cases where it's not found
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/')
def home():
    """
    A simple home route to check if the API is running.
    """
    return "Wine Quality Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predicts wine quality based on input chemical features.
    Expects a JSON payload with features.
    """
    if model is None:
        return jsonify({"error": "Model not loaded. Cannot make predictions."}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    # Expected features (in the correct order as used during training)
    expected_features = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]

    # Validate input data
    if not all(feature in data for feature in expected_features):
        missing_features = [f for f in expected_features if f not in data]
        return jsonify({
            "error": "Missing features in request body.",
            "missing": missing_features,
            "expected": expected_features
        }), 400

    try:
        # Create a DataFrame from the input data, ensuring column order
        input_df = pd.DataFrame([data], columns=expected_features)

        # Make prediction
        prediction = model.predict(input_df)
        # The model predicts a single quality score, so we take the first element
        predicted_quality = int(prediction[0])

        return jsonify({
            "predicted_quality": predicted_quality,
            "input_features": data
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    # For local development, run with Flask's built-in server
    # In production, Gunicorn will be used.
    app.run(debug=True, host='0.0.0.0', port=5000)

