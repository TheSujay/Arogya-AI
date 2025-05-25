import joblib

model = joblib.load('models/symptom_checker.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def predict_disease(text: str) -> str:
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return prediction[0]
