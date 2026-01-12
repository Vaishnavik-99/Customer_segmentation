import joblib
import numpy as np

model = joblib.load("../models/kmeans_model.pkl")
scaler = joblib.load("../models/scaler.pkl")

def predict_cluster(age, income, score):
    data = np.array([[age, income, score]])
    scaled = scaler.transform(data)
    cluster = model.predict(scaled)
    return cluster[0]

if __name__ == "__main__":
    print(predict_cluster(25, 40, 60))
