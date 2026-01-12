import joblib
import numpy as np

model = joblib.load("../models/kmeans_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

def predict_cluster(age, income, score):
    data = np.array([[age, income, score]])
    scaled = scaler.transform(data)
    return model.predict(scaled)[0]
