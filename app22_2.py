import os
import whisper
import warnings
import logging
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from celery import Celery
import redis

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Suppress FP16 warning from Whisper
warnings.filterwarnings("ignore", category=UserWarning, module="whisper")

# Flask app setup
app = Flask(__name__)
CORS(app)

# Celery configuration
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Whisper model with explicit CPU usage
try:
    model = whisper.load_model("base", device="cpu")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    raise

@celery.task(bind=True)
def transcribe_audio_task(self, file_path):
    try:
        logger.info(f"Processing file: {file_path}")
        result = model.transcribe(file_path)
        text = result["text"]
        os.remove(file_path)  # Clean up after transcription
        return {"transcription": text}
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        logger.error(traceback.format_exc())
        try:
            os.remove(file_path)  # Cleanup
        except:
            pass
        return {"error": "Transcription failed", "details": str(e)}

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    try:
        if "audio" not in request.files:
            return jsonify({"error": "No audio file provided"}), 400

        audio_file = request.files["audio"]
        if audio_file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, f"audio_{hash(audio_file.filename)}.wav")
        audio_file.save(file_path)
        logger.info(f"Saved audio file: {file_path}")
        
        # Add task to the queue
        task = transcribe_audio_task.apply_async(args=[file_path])
        return jsonify({"task_id": task.id, "status": "processing"})
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route("/result/<task_id>", methods=["GET"])
def get_transcription_result(task_id):
    task = transcribe_audio_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        return jsonify({"status": "pending"}), 202
    elif task.state == 'SUCCESS':
        return jsonify({"status": "completed", "result": task.result}), 200
    elif task.state == 'FAILURE':
        return jsonify({"status": "failed", "error": str(task.result)}), 500

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    from waitress import serve
    app.run(host="0.0.0.0", port=5000, debug=True)
