import asyncio
import websockets
import numpy as np
import soundfile as sf
import io
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os
import json
import google.generativeai as genai
from faster_whisper import WhisperModel

# Configuration
HOST = '0.0.0.0'
WS_PORT = 8080
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
# gemini_model = genai.GenerativeModel('gemini-pro')
gemini_model = genai.GenerativeModel('gemini-1.5-flash')
print("Gemini model initialized successfully!")

#=========================#
# WebSocket Server Section
#=========================#

async def handle_websocket_client(websocket):
    print(f"WebSocket client connected: {websocket.remote_address}")
    audio_buffer = b''
    
    try:
        async for message in websocket:
            # Append incoming audio data to buffer
            audio_buffer += message
            
            # Process when we have enough audio data
            if len(audio_buffer) >= SAMPLE_RATE * SAMPLE_WIDTH * CHUNK_DURATION_MS / 1000:
                audio_np = np.frombuffer(audio_buffer, dtype=np.int16)
                
                with io.BytesIO() as wav_io:
                    sf.write(wav_io, audio_np, SAMPLE_RATE, format='WAV')
                    wav_io.seek(0)
                    
                    # Use run_in_executor for non-blocking transcription
                    segments, _ = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: model.transcribe(wav_io, language="en")
                    )
                    
                    transcription = " ".join([segment.text for segment in segments])
                    print(f"[WebSocket] Transcription: {transcription}")
                    
                    # Send transcription back to client
                    if transcription.strip():  # Only send non-empty transcriptions
                        await websocket.send(transcription)
                
                # Clear buffer after processing
                audio_buffer = b''
                
    except websockets.exceptions.ConnectionClosed as e:
        print(f"WebSocket client disconnected: {e}")

async def run_websocket_server():
    print(f"Starting WebSocket server on {HOST}:{WS_PORT}...")
    
    async with websockets.serve(
        handle_websocket_client,
        HOST,
        WS_PORT,
        ping_interval=60,
        ping_timeout=60
    ):
        print(f"WebSocket server listening on {HOST}:{WS_PORT}")
        await asyncio.Future()  # Run forever

#===================#
# Flask API Section
#===================#

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def analyze_medical_conversation(transcription):
    """
    Analyze medical conversation using Gemini AI and extract structured information
    """
    prompt = f"""
    You are a medical AI assistant. Analyze the following medical conversation transcript and extract structured information in JSON format.

    Transcript: "{transcription}"

    Please analyze this medical conversation and provide a JSON response with the following structure:
    {{
        "summary": "A comprehensive paragraph summary of the entire conversation",
        "symptoms": ["list of patient symptoms mentioned"],
        "diagnosis": ["list of diagnoses mentioned by the doctor, in chronological order"],
        "medications": [
            {{
                "name": "medication name",
                "dosage": "dosage amount",
                "frequency": "frequency of administration"
            }}
        ],
        "follow_up": "follow-up instructions or next steps mentioned"
    }}

    Rules:
    1. Extract only information that is explicitly mentioned in the conversation
    2. If no information is available for a category, use empty arrays [] or empty strings ""
    3. For medications, include all three fields (name, dosage, frequency) if mentioned
    4. Be accurate and don't infer information not present in the transcript
    5. Return only valid JSON format

    Respond with only the JSON object, no additional text.
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        
        # Clean the response text to extract JSON
        response_text = response.text.strip()
        
        # Remove any markdown formatting if present
        if response_text.startswith('```json'):
            response_text = response_text[7:-3]
        elif response_text.startswith('```'):
            response_text = response_text[3:-3]
        
        # Parse the JSON response
        analysis = json.loads(response_text)
        
        return analysis
        
    except json.JSONDecodeError as e:
        print(f"[Gemini Analysis] JSON decode error: {str(e)}")
        print(f"[Gemini Analysis] Raw response: {response.text}")
        return {
            "summary": "Error: Could not parse analysis response",
            "symptoms": [],
            "diagnosis": [],
            "medications": [],
            "follow_up": ""
        }
    except Exception as e:
        print(f"[Gemini Analysis] Error: {str(e)}")
        return {
            "summary": "Error: Could not analyze conversation",
            "symptoms": [],
            "diagnosis": [],
            "medications": [],
            "follow_up": ""
        }

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
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        audio_file.save(temp_file.name)
        temp_filename = temp_file.name
    
    try:
        print(f"[Flask API - Analyse] Processing audio file: {temp_filename}")
        
        # Step 1: Transcribe the audio file
        segments, info = model.transcribe(temp_filename, language="en")
        transcription = " ".join([segment.text for segment in segments])
        
        print(f"[Flask API - Analyse] Transcription complete: {len(transcription)} characters")
        print(f"[Flask API - Analyse] Transcription: {transcription[:200]}...")
        
        if not transcription.strip():
            return jsonify({'error': 'No speech detected in audio file'}), 400
        
        # Step 2: Analyze the transcription using Gemini
        print("[Flask API - Analyse] Starting Gemini analysis...")
        analysis = analyze_medical_conversation(transcription)
        
        # Step 3: Combine transcription with analysis
        result = {
            "transcription": transcription,
            **analysis  # This unpacks the analysis dictionary
        }
        
        print(f"[Flask API - Analyse] Analysis complete")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"[Flask API - Analyse] Error during analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)

def run_flask_app():
    print(f"Starting Flask API server on {HOST}:{FLASK_PORT}...")
    app.run(host=HOST, port=FLASK_PORT, debug=False, use_reloader=False)

#===================#
# Main Entry Point
#===================#

if __name__ == "__main__":
    print("Starting Speech Transcription System...")
    
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.daemon = True  # This ensures the thread will exit when the main program exits
    flask_thread.start()
    
    # Run the WebSocket server in the main thread
    asyncio.run(run_websocket_server())
