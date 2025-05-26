import os
import joblib
import pandas as pd
import pytest
from app.app import app # Import the Flask app for testing API endpoints

# Define paths relative to the test file
MODEL_PATH = 'model/wine_quality_model.pkl'
DATA_PATH = 'data/winequality-red.csv'

# Fixture to load the model once for all tests
@pytest.fixture(scope='module')
def trained_model():
    """Loads the trained model for testing."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model file not found at {MODEL_PATH}. Please train the model first.")
    return joblib.load(MODEL_PATH)

# Fixture to load a sample of the dataset for testing
@pytest.fixture(scope='module')
def sample_data():
    """Loads a sample of the wine quality dataset for testing."""
    if not os.path.exists(DATA_PATH):
        pytest.skip(f"Data file not found at {DATA_PATH}. Please ensure 'winequality-red.csv' is in the 'data/' directory.")
   # df = pd.read_csv(DATA_PATH)
    df = pd.read_csv(DATA_PATH, delimiter=';')
    # Select a few rows for testing, excluding the 'quality' column
    return df.drop('quality', axis=1).head(5)

# Fixture for Flask test client
@pytest.fixture
def client():
    """Configures the Flask app for testing and returns a test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

# --- Model Tests ---

def test_model_loading(trained_model):
    """Test that the model loads successfully."""
    assert trained_model is not None
    assert hasattr(trained_model, 'predict')

def test_model_prediction_output_type(trained_model, sample_data):
    """Test that the model prediction returns an integer type."""
    if sample_data.empty:
        pytest.skip("Sample data is empty, cannot perform prediction test.")
    predictions = trained_model.predict(sample_data)
    # assert isinstance(predictions[0], (int, float)) # Scikit-learn can return float for some models, int for others
    assert isinstance(int(predictions[0]), (int, float))

def test_model_prediction_on_sample_data(trained_model, sample_data):
    """Test that the model can make predictions on sample data."""
    if sample_data.empty:
        pytest.skip("Sample data is empty, cannot perform prediction test.")
    predictions = trained_model.predict(sample_data)
    assert len(predictions) == len(sample_data)
    # Basic check for reasonable quality scores (e.g., between 3 and 8)
    assert all(3 <= p <= 8 for p in predictions)

# --- API Tests ---

def test_home_endpoint(client):
    """Test the / endpoint."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"Wine Quality Prediction API is running!" in response.data

def test_predict_endpoint_success(client, sample_data):
    """Test the /predict endpoint with valid data."""
    if sample_data.empty:
        pytest.skip("Sample data is empty, cannot perform API prediction test.")

    # Convert the first sample data row to a dictionary for JSON payload
    test_input = sample_data.iloc[0].to_dict()
    response = client.post('/predict', json=test_input)

    assert response.status_code == 200
    data = response.get_json()
    assert "predicted_quality" in data
    assert isinstance(data["predicted_quality"], int)
    assert 3 <= data["predicted_quality"] <= 8 # Check for reasonable range

def test_predict_endpoint_missing_features(client):
    """Test the /predict endpoint with missing features."""
    invalid_input = {
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        # Missing other features
    }
    response = client.post('/predict', json=invalid_input)

    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "Missing features" in data["error"]
    assert "missing" in data
    assert "expected" in data

def test_predict_endpoint_non_json_request(client):
    """Test the /predict endpoint with a non-JSON request."""
    response = client.post('/predict', data="plain text data")
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "Request must be JSON" in data["error"]

# Optional: Add a test for model not loaded scenario (requires modifying app.py for testing)
# For now, we assume model is loaded or the test will skip if not found.
