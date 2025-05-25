import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Dummy symptom data
data = {
    'fever': [1, 0, 1, 0],
    'cough': [1, 1, 0, 0],
    'headache': [1, 0, 1, 0],
    'fatigue': [1, 1, 1, 0],
    'diagnosis': ['Flu', 'Cold', 'Migraine', 'Healthy']
}

df = pd.DataFrame(data)
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'symptom_checker.pkl')
print("✅ Model saved as symptom_checker.pkl")
