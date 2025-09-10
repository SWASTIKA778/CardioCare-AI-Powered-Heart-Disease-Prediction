from flask import Flask, request, jsonify
import pandas as pd
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os

app = Flask(__name__)

MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'


def load_data():

    
    data = pd.read_csv('heart.csv')
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(StringIO(data), names=columns, na_values='?')
    df = df.dropna()
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    return df

def train_model():
    df = load_data()
    features = ['age', 'chol', 'trestbps', 'fbs']
    X = df[features]
    y = df['target']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression()
    model.fit(X_scaled, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    train_model()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        age = float(data['age'])
        cholesterol = float(data['cholesterol'])
        bloodPressure = float(data['bloodPressure'])
        sugar = float(data['sugar'])
    except (KeyError, ValueError, TypeError):
        return jsonify({'error': 'Invalid input data'}), 400

    input_data = [[age, cholesterol, bloodPressure, sugar]]
    input_scaled = scaler.transform(input_data)
    prob = model.predict_proba(input_scaled)[0][1]
    if prob < 0.3:
        prediction = 'Low Risk'
    elif prob < 0.7:
        prediction = 'Medium Risk'
    else:
        prediction = 'High Risk'

    return jsonify({'prediction': prediction, 'probability': round(prob, 2)})

if __name__ == '__main__':
    app.run(debug=True)
