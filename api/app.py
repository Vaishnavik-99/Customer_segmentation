from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("../models/kmeans_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

@app.get("/")
def home():
    return {"message": "Customer Segmentation API"}

@app.post("/predict")
def predict(age: int, income: float, score: float):
    data = np.array([[age, income, score]])
    scaled = scaler.transform(data)
    cluster = int(model.predict(scaled)[0])

    return {"cluster": cluster}
