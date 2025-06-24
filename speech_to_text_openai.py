import numpy as np
import soundfile as sf
import io
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import tempfile
import os
import json
from openai import AzureOpenAI
from faster_whisper import WhisperModel

# Configuration
HOST = '0.0.0.0'
FLASK_PORT = 5000
MODEL_SIZE = "medium.en"
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1
CHUNK_DURATION_MS = 1000

# Azure OpenAI Configuration - Updated with correct endpoint and deployment
AZURE_OPENAI_ENDPOINT = "https://scrib-mc9xkslj-eastus2.openai.azure.com/"
AZURE_OPENAI_API_KEY = "5NdvOKi12atZUzZcUXafW3OmeyUhNslWx5e1QA8yVWlVZ09OcyaqJQQJ99BFACHYHv6XJ3w3AAAAACOGovPz"
AZURE_OPENAI_API_VERSION = "2025-01-01-preview"
AZURE_DEPLOYMENT_NAME = "o4-mini"

print(f"Initializing Whisper model: {MODEL_SIZE}")
# Initialize the model once for both services
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
print("Model loaded successfully!")

# Initialize Azure OpenAI client with updated configuration
azure_client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION
)
print("Azure OpenAI client initialized successfully!")

#===================#
# Flask API Section
#===================#

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def analyze_with_template(transcription, template):
    """
    Analyze medical conversation using Azure OpenAI with provided template
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
        # Updated message format with developer role and proper structure
        messages = [
            {
                "role": "developer",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a medical AI assistant specialized in analyzing medical conversations and creating structured reports."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
        
        response = azure_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=messages,
            max_completion_tokens=2000,
            stream=False
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"[Azure OpenAI Analysis] Error: {str(e)}")
        return f"Error: Could not analyze conversation - {str(e)}"

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
        
        # Get the complete analysis
        analysis_result = analyze_with_template(transcription, template)
        
        # Return simple JSON response like the second code
        return jsonify({'data': analysis_result})
    
    except Exception as e:
        print(f"[Flask API - Analyse] Error during analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/test-azure', methods=['GET'])
def test_azure():
    """
    Test endpoint to verify Azure OpenAI connection
    """
    try:
        messages = [
            {
                "role": "developer",
                "content": [
                    {
                        "type": "text",
                        "text": "You are an AI assistant that helps people find information."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Say hello and confirm you are working properly."
                    }
                ]
            }
        ]
        
        response = azure_client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=messages,
            max_completion_tokens=1000,
            stream=False
        )
        
        return jsonify({
            'status': 'success',
            'response': response.choices[0].message.content.strip(),
            'model': AZURE_DEPLOYMENT_NAME,
            'endpoint': AZURE_OPENAI_ENDPOINT
        })
        
    except Exception as e:
        print(f"[Test Azure] Error: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'model': AZURE_DEPLOYMENT_NAME,
            'endpoint': AZURE_OPENAI_ENDPOINT
        }), 500

#===================#
# Main Entry Point
#===================#

if __name__ == "__main__":
    print("Starting Speech Transcription API...")
    print(f"Azure OpenAI Endpoint: {AZURE_OPENAI_ENDPOINT}")
    print(f"Deployment Name: {AZURE_DEPLOYMENT_NAME}")
    print(f"API Version: {AZURE_OPENAI_API_VERSION}")
    print(f"Flask API server starting on {HOST}:{FLASK_PORT}...")
    app.run(host=HOST, port=FLASK_PORT, debug=False)
