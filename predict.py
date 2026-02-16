import joblib
import numpy as np

# Load model and scalers
model = joblib.load("fraud_detection_model.pkl")
amount_scaler = joblib.load("amount_scaler.pkl")
time_scaler = joblib.load("time_scaler.pkl")

def predict_transaction(features):
    """
    features: list of 30 values 
    [Amount, Time, V1, V2, ..., V28]
    """

    amount_scaled = amount_scaler.transform([[features[0]]])[0][0]
    time_scaled = time_scaler.transform([[features[1]]])[0][0]

    final_features = [amount_scaled, time_scaled] + features[2:]
    final_features = np.array(final_features).reshape(1, -1)

    prediction = model.predict(final_features)[0]
    probability = model.predict_proba(final_features)[0][1]

    return prediction, probability


if __name__ == "__main__":
    sample = [100, 50000] + [0]*28

    pred, prob = predict_transaction(sample)

    print("Prediction:", "Fraud" if pred == 1 else "Normal")
    print("Fraud Probability:", round(prob, 4))
