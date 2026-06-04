from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load('habit_model.pkl')

@app.route('/')
def home():
    return jsonify({'status': 'Mwendo API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [
        float(data['study_hours_per_day']),
        float(data['sleep_hours']),
        float(data['phone_usage_hours']),
        float(data['social_media_hours']),
        float(data['exercise_minutes']),
        float(data['stress_level']),
        float(data['breaks_per_day'])
    ]
    prediction = model.predict([features])[0]
    return jsonify({'productivity_score': round(float(prediction), 1)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)