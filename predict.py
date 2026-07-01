import joblib
import numpy as np
from feature_extraction import extract_features

model = joblib.load("emotion_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def pred_emotion(audio_path):
    features=extract_features(audio_path)
    features=features.reshape(1,-1)
    features=scaler.transform(features)

    prediction=model.predict(features)
    emotion=label_encoder.inverse_transform(prediction)
    probab=model.predict_proba(features)
    confidence=np.max(probab)
    return emotion[0],confidence

if __name__ == "__main__":

    audio_path = r"Actor_01\03-01-03-01-01-01-01.wav"

    emotion,confidence = pred_emotion(audio_path)

    print("Predicted Emotion:", emotion)
    print("Confidence :", confidence)
