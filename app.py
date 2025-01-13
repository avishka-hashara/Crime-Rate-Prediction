from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('crime_rate_model.pkl')

@app.route('/')
def home():
    return "Crime Rate Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON data from the request
        data = request.get_json()

        # Convert the input data to a NumPy array
        features = np.array([data['features']])

        # Make a prediction
        prediction = model.predict(features)

        # Return the result as JSON
        return jsonify({'crime_rate_prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
