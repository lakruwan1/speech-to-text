import numpy as np
import soundfile as sf
import io
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import tempfile
import os
import json
import google.generativeai as genai
from faster_whisper import WhisperModel

# Configuration
HOST = '0.0.0.0'
FLASK_PORT = 5000
MODEL_SIZE = "medium.en"
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1
CHUNK_DURATION_MS = 1000

# Gemini API Configuration
GEMINI_API_KEY = "AIzaSyCzQsnMap61JhNozh9CQGYa8AAFsOR3yAM"
genai.configure(api_key=GEMINI_API_KEY)

print(f"Initializing Whisper model: {MODEL_SIZE}")
# Initialize the model once for both services
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
print("Model loaded successfully!")

# Initialize Gemini model
gemini_model = genai.GenerativeModel('gemini-1.5-flash')
print("Gemini model initialized successfully!")

#===================#
# Flask API Section
#===================#

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def analyze_with_template(transcription, template):
    """
    Analyze medical conversation using Gemini AI with provided template
    """
    prompt = f"""
    You are a medical AI assistant. Analyze the following medical conversation transcript and create a structured medical report using the provided template format.

    Transcript: "{transcription}"

    Template Format:
    {template}

    Instructions:
    1. Use the exact template structure provided
    2. Only include information that is explicitly mentioned in the transcript
    3. If information for any section is not mentioned in the transcript, leave that section blank (do not include placeholder text)
    4. Make all section titles bold using **Title** format
    5. Use proper line breaks for readability
    6. Be accurate and don't infer information not present in the transcript
    7. Follow the template structure exactly as provided

    Create the medical report based on the transcript and template:
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
        
    except Exception as e:
        print(f"[Gemini Analysis] Error: {str(e)}")
        return f"Error: Could not analyze conversation - {str(e)}"

def stream_analysis(transcription, template):
    """
    Stream the analysis response from Gemini AI
    """
    prompt = f"""
    You are a medical AI assistant. Analyze the following medical conversation transcript and create a structured medical report using the provided template format.

    Transcript: "{transcription}"

    Template Format:
    {template}

    Instructions:
    1. Use the exact template structure provided
    2. Only include information that is explicitly mentioned in the transcript
    3. If information for any section is not mentioned in the transcript, leave that section blank (do not include placeholder text)
    4. Make all section titles bold using **Title** format
    5. Use proper line breaks for readability
    6. Be accurate and don't infer information not present in the transcript
    7. Follow the template structure exactly as provided

    Create the medical report based on the transcript and template:
    """
    
    try:
        response = gemini_model.generate_content(prompt, stream=True)
        
        for chunk in response:
            if chunk.text:
                # Format the response as JSON for streaming
                data = {"data": chunk.text}
                yield f"data: {json.dumps(data)}\n\n"
                
    except Exception as e:
        error_data = {"data": f"Error: Could not analyze conversation - {str(e)}"}
        yield f"data: {json.dumps(error_data)}\n\n"

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        audio_file.save(temp_file.name)
        temp_filename = temp_file.name
    
    try:
        print(f"[Flask API] Processing full audio file: {temp_filename}")
        
        # Transcribe the audio file
        segments, info = model.transcribe(temp_filename, language="en")
        
        # Combine all segments into a single transcription
        transcription = " ".join([segment.text for segment in segments])
        
        print(f"[Flask API] Transcription complete: {len(transcription)} characters")
        
        return jsonify({
            'transcription': transcription,
            'language': info.language,
            'duration': info.duration
        })
    
    except Exception as e:
        print(f"[Flask API] Error during transcription: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)

@app.route('/analyse', methods=['POST'])
def analyse():
    try:
        # Get data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        transcription = data.get('transcription', '')
        template = data.get('template', '')
        
        if not transcription:
            return jsonify({'error': 'No transcription provided'}), 400
        
        if not template:
            return jsonify({'error': 'No template provided'}), 400
        
        print(f"[Flask API - Analyse] Starting analysis...")
        print(f"[Flask API - Analyse] Transcription length: {len(transcription)} characters")
        print(f"[Flask API - Analyse] Template length: {len(template)} characters")
        
        # Return streaming response
        return Response(
            stream_analysis(transcription, template),
            mimetype='text/plain',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
            }
        )
    
    except Exception as e:
        print(f"[Flask API - Analyse] Error during analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

#===================#
# Main Entry Point
#===================#

if __name__ == "__main__":
    print("Starting Speech Transcription API...")
    print(f"Flask API server starting on {HOST}:{FLASK_PORT}...")
    app.run(host=HOST, port=FLASK_PORT, debug=False)
