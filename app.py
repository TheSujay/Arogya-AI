from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import re
import traceback

app = Flask(__name__)
CORS(app, supports_credentials=True, origins=["http://localhost:5173", "http://127.0.0.1:5173"])

# === Static knowledge base for auto-suggestions ===
medical_knowledge = {
    "flu": {
        "symptoms": ["fever", "chills", "muscle aches", "cough", "congestion", "runny nose"],
        "description": "A common viral infection that can be deadly for high-risk groups."
    },
    "migraine": {
        "symptoms": ["headache", "nausea", "sensitivity to light", "blurred vision"],
        "description": "A neurological condition that causes severe headaches and other symptoms."
    },
    "covid-19": {
        "symptoms": ["fever", "cough", "tiredness", "loss of taste or smell", "difficulty breathing"],
        "description": "A viral respiratory illness caused by the coronavirus SARS-CoV-2."
    },
    "diabetes": {
        "symptoms": ["increased thirst", "frequent urination", "extreme hunger", "fatigue"],
        "description": "A chronic condition that affects the way your body processes blood sugar."
    }
}

# === API Keys and Endpoints ===
GROQ_API_KEY = "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"
GROQ_MODEL = "llama3-8b-8192"

GEMINI_API_KEY = "AIzaSyBi2VKPUmq9RQsOrZe5HN-DCZvPXDWTZig"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

# === Utility: Parse LLM Output into JSON ===
def parse_prediction_to_json(content):
    conditions = []
    follow_up_questions = []

    # Split into conditions and questions
    parts = content.split("### 🧠 Follow-Up Questions")
    condition_block = parts[0].strip()
    question_block = parts[1].strip() if len(parts) > 1 else ""

    # Match each condition section
    condition_sections = re.split(r"### 🩺 Condition \d+[:：]?", condition_block)
    for section in condition_sections:
        if not section.strip():
            continue

        name_match = re.search(r"(Condition \d+:)?\s*(.*)", section)
        condition_name = name_match.group(2).strip().split('\n')[0] if name_match else "N/A"

        description = re.search(r"Description[:：]?\s*(.*)", section)
        treatment = re.search(r"Treatment Advice[:：]?\s*(.*)", section)
        risk = re.search(r"Risk Level[:：]?\s*(.*)", section)
        specialist = re.search(r"Specialist to Consult[:：]?\s*(.*)", section)

        conditions.append({
            "condition": condition_name,
            "description": f"📝 {description.group(1).strip() if description else 'N/A'}\n"
                           f"💊Treatment:   {treatment.group(1).strip() if treatment else 'N/A'}\n"
                           f"⚠️Risk Level:  {risk.group(1).strip() if risk else 'N/A'}\n"
                           f"👨‍⚕️Specialist:  {specialist.group(1).strip() if specialist else 'N/A'}"
        })

    # Parse follow-up questions
    question_lines = question_block.strip().split("\n")
    for line in question_lines:
        q_match = re.match(r"\d+\.\s*(.*)", line.strip())
        if q_match:
            follow_up_questions.append(q_match.group(1).strip())

    return conditions, follow_up_questions


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptoms = data.get("symptoms", "").strip()

    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400

    prompt = f"""
You are a professional medical assistant.

A patient describes the following symptoms: "{symptoms}"

Please return exactly 3 possible medical conditions in the following format (and don't leave any field blank, use 'N/A' if unsure):

### 🩺 Condition 1: <Condition Name>
- 📝 Description: <brief explanation>
- 💊 Treatment Advice: <suggested treatment>
- ⚠️ Risk Level: Low / Moderate / High
- 👨‍⚕️ Specialist to Consult: <type of doctor>

### 🩺 Condition 2: <Condition Name>
- 📝 Description: ...
- 💊 Treatment Advice: ...
- ⚠️ Risk Level: ...
- 👨‍⚕️ Specialist to Consult: ...

### 🩺 Condition 3: <Condition Name>
- 📝 Description: ...
- 💊 Treatment Advice: ...
- ⚠️ Risk Level: ...
- 👨‍⚕️ Specialist to Consult: ...

Then write:

### 🧠 Follow-Up Questions
1. What are your other symptoms?
2. How long have you had them?
3. Any recent travel or contact with sick people?

⚠️ Format strictly like this. Do not skip or invent format. Every field must be filled or marked as 'N/A'.
"""

    try:
        groq_response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": GROQ_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.5
            }
        )
        groq_response.raise_for_status()
        content = groq_response.json()["choices"][0]["message"]["content"]
        print("✅ GROQ Output:\n", content)
    except Exception as e:
        print("❌ GROQ failed:", e)
        traceback.print_exc()
        try:
            gemini_response = requests.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]}
            )
            gemini_response.raise_for_status()
            content = gemini_response.json()["candidates"][0]["content"]["parts"][0]["text"]
            print("✅ Gemini Output:\n", content)
        except Exception as ge:
            print("❌ Gemini failed:", ge)
            traceback.print_exc()
            return jsonify({
                "prediction": [{"condition": "Error", "description": "Both GROQ and Gemini failed."}],
                "questions": []
            }), 500

    # ✅ Convert to structured JSON
    conditions, questions = parse_prediction_to_json(content)

    return jsonify({
        "prediction": conditions,
        "questions": questions
    }), 200


@app.route("/suggest", methods=["POST"])
def suggest():
    data = request.get_json()
    query = data.get("query", "").lower()

    if not query:
        return jsonify({"suggestions": []})

    all_keywords = set()
    for disease, info in medical_knowledge.items():
        all_keywords.add(disease)
        all_keywords.update(info["symptoms"])

    suggestions = [item for item in all_keywords if query in item.lower()]
    return jsonify({"suggestions": suggestions[:10]})


def run_server():
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    run_server()
