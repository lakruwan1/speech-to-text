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
import time

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

def analyze_medical_conversation_streaming(transcription):
    """
    Analyze medical conversation using Gemini AI with streaming response
    """
    prompt = f"""
    You are a medical AI assistant. Analyze the following medical conversation transcript and extract structured information.

    Transcript: "{transcription}"

    Please analyze this medical conversation and provide a comprehensive analysis with the following sections:

    **Summary:**
    Provide a comprehensive paragraph summary of the entire conversation.

    **Symptoms:**
    List all patient symptoms mentioned in the conversation.

    **Diagnosis:**
    List all diagnoses mentioned by the doctor, in chronological order.

    **Medications:**
    For each medication mentioned, provide:
    - Name of medication
    - Dosage amount
    - Frequency of administration

    **Follow-up Instructions:**
    Provide any follow-up instructions or next steps mentioned.

    Rules:
    1. Extract only information that is explicitly mentioned in the conversation
    2. If no information is available for a category, mention "None mentioned"
    3. Use **bold formatting** for section titles
    4. Use clear line breaks between sections
    5. Be accurate and don't infer information not present in the transcript
    """
    
    try:
        # Generate content with streaming
        response = gemini_model.generate_content(
            prompt,
            stream=True,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=2048,
            )
        )
        
        for chunk in response:
            if chunk.text:
                # Format the response in the required JSON format
                yield f"data: {json.dumps({'data': chunk.text})}\n\n"
                time.sleep(0.05)  # Small delay for better streaming effect
        
    except Exception as e:
        print(f"[Gemini Analysis] Error: {str(e)}")
        error_msg = f"Error: Could not analyze conversation - {str(e)}"
        yield f"data: {json.dumps({'data': error_msg})}\n\n"

def analyze_medical_conversation(transcription):
    """
    Analyze medical conversation using Gemini AI and extract structured information (non-streaming)
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

@app.route('/analyse-stream', methods=['POST'])
def analyse_stream():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        audio_file.save(temp_file.name)
        temp_filename = temp_file.name
    
    try:
        print(f"[Flask API - Stream Analyse] Processing audio file: {temp_filename}")
        
        # Step 1: Transcribe the audio file
        segments, info = model.transcribe(temp_filename, language="en")
        transcription = " ".join([segment.text for segment in segments])
        
        print(f"[Flask API - Stream Analyse] Transcription complete: {len(transcription)} characters")
        
        if not transcription.strip():
            def error_stream():
                yield f"data: {json.dumps({'data': 'Error: No speech detected in audio file'})}\n\n"
            
            return Response(error_stream(), mimetype='text/plain')
        
        # First send the transcription
        def stream_response():
            yield f"data: {json.dumps({'data': f'**Transcription:**\\n{transcription}\\n\\n'})}\n\n"
            time.sleep(0.5)
            yield f"data: {json.dumps({'data': '**Analysis:**\\n\\n'})}\n\n"
            time.sleep(0.3)
            
            # Then stream the analysis
            for chunk in analyze_medical_conversation_streaming(transcription):
                yield chunk
        
        return Response(stream_response(), mimetype='text/plain')
    
    except Exception as e:
        print(f"[Flask API - Stream Analyse] Error during analysis: {str(e)}")
        def error_stream():
            yield f"data: {json.dumps({'data': f'Error: {str(e)}'})}\n\n"
        
        return Response(error_stream(), mimetype='text/plain')
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)

@app.route('/stream-test', methods=['POST'])
def stream_test():
    """Test endpoint for streaming functionality"""
    data = request.get_json()
    text = data.get('text', 'Hello, this is a streaming test!')
    
    def generate():
        for char in text:
            yield f"data: {json.dumps({'data': char})}\n\n"
            time.sleep(0.1)  # Simulate typing delay
    
    return Response(generate(), mimetype='text/plain')

#===================#
# Main Entry Point
#===================#

if __name__ == "__main__":
    print("Starting Speech Transcription API...")
    print(f"Flask API server starting on {HOST}:{FLASK_PORT}...")
    app.run(host=HOST, port=FLASK_PORT, debug=False)
