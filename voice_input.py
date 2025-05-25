from flask import Flask, request, jsonify
from flask_cors import CORS
import speech_recognition as sr

app = Flask(__name__)
CORS(app)

@app.route('/voice-predict', methods=['GET'])
def voice_predict():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    try:
        with mic as source:
            print("🎙️ Listening for symptoms...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        text = recognizer.recognize_google(audio)
        print(f"✅ Recognized: {text}")

        # You can optionally add AI logic here to predict a condition from `text`.
        return jsonify({'prediction': text})
    
    except sr.UnknownValueError:
        return jsonify({'error': "Sorry, I could not understand the audio."}), 400
    except sr.RequestError:
        return jsonify({'error': "Speech Recognition API unavailable."}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
