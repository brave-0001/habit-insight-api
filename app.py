from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'habit_model.pkl'

def train_model():
    from sklearn.datasets import make_regression
    # Synthetic training data matching our features
    np.random.seed(42)
    n = 2000
    study = np.random.uniform(0, 12, n)
    sleep = np.random.uniform(3, 12, n)
    phone = np.random.uniform(0, 12, n)
    social = np.random.uniform(0, 8, n)
    exercise = np.random.uniform(0, 120, n)
    stress = np.random.uniform(1, 10, n)
    breaks = np.random.uniform(0, 10, n)

    X = np.column_stack([study, sleep, phone, social, exercise, stress, breaks])
    y = (
        study * 4 +
        sleep * 3 +
        (12 - phone) * 2 +
        (8 - social) * 2 +
        exercise * 0.15 +
        (10 - stress) * 2 +
        breaks * 1.5
    )
    y = np.clip((y / y.max()) * 100, 0, 100)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    print("Training model...")
    model = train_model()
    print("Model ready!")

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