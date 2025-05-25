import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

symptoms = [
    "fever cough tiredness", "headache nausea",
    "chest pain breathlessness", "rash itchiness"
]
diseases = ["flu", "migraine", "heart disease", "allergy"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(symptoms)

model = RandomForestClassifier()
model.fit(X, diseases)

joblib.dump(model, 'models/symptom_checker.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
print("✅ Model and vectorizer saved successfully.")
