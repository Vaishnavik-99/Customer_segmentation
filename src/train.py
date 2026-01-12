import joblib
from sklearn.cluster import KMeans
from preprocess import load_and_preprocess

X_scaled, scaler = load_and_preprocess("../data/customers.csv")

model = KMeans(n_clusters=4, random_state=42)
model.fit(X_scaled)

joblib.dump(model, "../models/kmeans_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

print("Model trained")
