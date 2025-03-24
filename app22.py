from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os
import warnings

# Suppress FP16 warning from Whisper
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

app = Flask(__name__)
CORS(app)

# Load Whisper model with explicit CPU usage
model = whisper.load_model("base", device="cpu")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    file_path = os.path.join(UPLOAD_FOLDER, "audio.wav")
    audio_file.save(file_path)

    # Transcribe audio
    result = model.transcribe(file_path)
    text = result["text"]

    return jsonify({"transcription": text})

if __name__ == "__main__":
    app.run(debug=True)
