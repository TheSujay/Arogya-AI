from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
import os  # ✅ Needed for environment variable PORT

app = Flask(__name__)
CORS(app)

client = Groq(api_key="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")

medical_knowledge = {
    "Deep vein thrombosis": ["leg pain", "swelling", "redness"],
    "Muscle strain": ["leg pain", "stiffness"],
    "Peripheral artery disease": ["leg pain", "numbness", "coldness"],
    "Food poisoning": ["nausea", "vomiting", "diarrhea"],
    "Gastroenteritis": ["nausea", "fever", "diarrhea"],
    "Pregnancy": ["nausea", "fatigue"],
    "Flu": ["fever", "cough", "fatigue"],
    "COVID-19": ["fever", "cough", "loss of taste"],
    "Malaria": ["fever", "chills", "sweating"]
}

def filter_with_knowledge_base(symptoms, ai_predictions):
    symptoms = [s.strip().lower() for s in symptoms.split(",")]
    valid_diseases = []
    for disease in ai_predictions:
        if disease in medical_knowledge:
            disease_symptoms = [sym.lower() for sym in medical_knowledge[disease]]
            if any(symptom in symptoms for symptom in disease_symptoms):
                valid_diseases.append(disease)
    return valid_diseases if valid_diseases else ai_predictions

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data.get('symptoms', '').lower()
    if not symptoms:
        return jsonify({'error': 'Symptoms required'}), 400

    prompt = (
        f"You are a medical expert AI. Given these symptoms: {symptoms}, "
        "list the top 5 most likely diseases or medical conditions."
    )

    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # ✅ Groq-supported model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )

        ai_prediction_text = response.choices[0].message.content
        ai_predictions = [line.strip("-• ").strip() for line in ai_prediction_text.split('\n') if line]

        filtered_predictions = filter_with_knowledge_base(symptoms, ai_predictions)

        return jsonify({'prediction': filtered_predictions})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ✅ Fix for Render deployment
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render sets PORT dynamically
    app.run(host="0.0.0.0", port=port)