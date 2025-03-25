import flask
from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import os
import warnings
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Suppress FP16 warning from Whisper
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

app = Flask(__name__)
CORS(app)

# Load Whisper model with explicit CPU usage
try:
    model = whisper.load_model("base", device="cpu")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    raise

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    try:
        # Check if audio file is present
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio"]
        
        # Check if filename is empty
        if audio_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Generate a unique filename
        file_path = os.path.join(UPLOAD_FOLDER, "audio_" + str(hash(audio_file.filename)) + ".wav")
        
        # Save the file
        audio_file.save(file_path)
        
        # Log file details
        logger.info(f"Saved audio file: {file_path}")
        logger.info(f"File size: {os.path.getsize(file_path)} bytes")

        # Transcribe audio
        try:
            result = model.transcribe(file_path)
            text = result["text"]
            
            # Clean up the uploaded file
            os.remove(file_path)
            
            return jsonify({"transcription": text})
        
        except Exception as transcribe_error:
            logger.error(f"Transcription error: {transcribe_error}")
            logger.error(traceback.format_exc())
            
            # Attempt to remove the file even if transcription fails
            try:
                os.remove(file_path)
            except:
                pass
            
            return jsonify({
                "error": "Transcription failed", 
                "details": str(transcribe_error)
            }), 500

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)