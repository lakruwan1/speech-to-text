# multi_user_transcriber.py
# Modified to support multiple clients (up to 5 concurrent users)
# This script sets up a WebSocket server for real-time audio transcription using the Whisper model.
# It also provides a Flask API for batch transcription of audio files.

import asyncio
import websockets
import numpy as np
import soundfile as sf
import io
import threading
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import os
from faster_whisper import WhisperModel
import time
from collections import defaultdict

# Configuration
HOST = '0.0.0.0'
WS_PORT = 8080
FLASK_PORT = 5000
MODEL_SIZE = "medium.en"
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1
CHUNK_DURATION_MS = 1000
MAX_USERS = 5  # Maximum number of concurrent users

print(f"Initializing Whisper model: {MODEL_SIZE}")
# Initialize the model once for both services
model = WhisperModel(MODEL_SIZE, device="cuda", compute_type="float16")
print("Model loaded successfully!")

# Client management
active_clients = {}  # Dictionary to track active clients and their buffers
client_lock = threading.Lock()  # Lock for thread-safe client management
model_lock = threading.Lock()  # Lock for thread-safe model access

#=========================#
# WebSocket Server Section
#=========================#

async def handle_websocket_client(websocket):
    # Generate a unique client ID
    client_id = str(uuid.uuid4())
    client_address = websocket.remote_address
    
    # Check if we can accept more clients
    with client_lock:
        if len(active_clients) >= MAX_USERS:
            print(f"Maximum users reached. Rejecting client: {client_address}")
            await websocket.send("ERROR: Server at maximum capacity. Please try again later.")
            await websocket.close(1008, "Server at maximum capacity")
            return
        
        # Add client to active clients
        active_clients[client_id] = {
            'websocket': websocket,
            'audio_buffer': b'',
            'last_activity': time.time(),
            'address': client_address
        }
    
    print(f"WebSocket client connected: {client_address} (Client ID: {client_id})")
    print(f"Active clients: {len(active_clients)}/{MAX_USERS}")
    
    try:
        # Send confirmation of connection
        await websocket.send(f"Connected to transcription service. Your session ID: {client_id}")
        
        async for message in websocket:
            # Update client's last activity timestamp
            with client_lock:
                if client_id in active_clients:
                    active_clients[client_id]['last_activity'] = time.time()
                    active_clients[client_id]['audio_buffer'] += message
                    
                    # Get the current buffer for processing
                    audio_buffer = active_clients[client_id]['audio_buffer']
                    
                    # Process when we have enough audio data
                    if len(audio_buffer) >= SAMPLE_RATE * SAMPLE_WIDTH * CHUNK_DURATION_MS / 1000:
                        # Process audio in a non-blocking way
                        asyncio.create_task(process_audio(client_id, audio_buffer))
                        # Clear buffer after dispatching for processing
                        active_clients[client_id]['audio_buffer'] = b''
    
    except websockets.exceptions.ConnectionClosed as e:
        print(f"WebSocket client disconnected: {client_address} (Client ID: {client_id}) - {e}")
    finally:
        # Remove client from active clients
        with client_lock:
            if client_id in active_clients:
                del active_clients[client_id]
                print(f"Removed client {client_id}. Active clients: {len(active_clients)}/{MAX_USERS}")

async def process_audio(client_id, audio_buffer):
    """Process audio data for a specific client"""
    try:
        with client_lock:
            if client_id not in active_clients:
                # Client disconnected while we were processing
                return
                
            websocket = active_clients[client_id]['websocket']
        
        # Convert audio buffer to numpy array
        audio_np = np.frombuffer(audio_buffer, dtype=np.int16)
        
        with io.BytesIO() as wav_io:
            sf.write(wav_io, audio_np, SAMPLE_RATE, format='WAV')
            wav_io.seek(0)
            
            # Acquire model lock to ensure one transcription at a time
            with model_lock:
                # Use run_in_executor for non-blocking transcription
                segments, _ = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: model.transcribe(wav_io, language="en")
                )
                
                transcription = " ".join([segment.text for segment in segments])
            
            print(f"[WebSocket] Client {client_id}: {transcription}")
            
            # Send transcription back to client if still connected
            with client_lock:
                if client_id in active_clients and transcription.strip():
                    await websocket.send(transcription)
    
    except Exception as e:
        print(f"Error processing audio for client {client_id}: {str(e)}")

async def cleanup_inactive_clients():
    """Periodically check and remove inactive clients"""
    while True:
        try:
            current_time = time.time()
            clients_to_remove = []
            
            # Check for inactive clients (no activity for 5 minutes)
            with client_lock:
                for client_id, client_data in active_clients.items():
                    if current_time - client_data['last_activity'] > 300:  # 5 minutes = 300 seconds
                        clients_to_remove.append(client_id)
            
            # Remove inactive clients
            for client_id in clients_to_remove:
                with client_lock:
                    if client_id in active_clients:
                        try:
                            await active_clients[client_id]['websocket'].close(1000, "Session timeout due to inactivity")
                        except:
                            pass
                        del active_clients[client_id]
                        print(f"Removed inactive client {client_id}. Active clients: {len(active_clients)}/{MAX_USERS}")
            
            # Run cleanup every 60 seconds
            await asyncio.sleep(60)
        
        except Exception as e:
            print(f"Error in cleanup task: {str(e)}")
            await asyncio.sleep(60)

async def run_websocket_server():
    print(f"Starting WebSocket server on {HOST}:{WS_PORT}...")
    
    # Start the cleanup task
    asyncio.create_task(cleanup_inactive_clients())
    
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

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    # Check if we've reached maximum concurrent API requests
    with client_lock:
        # Count ongoing API requests (this is approximate but helps manage load)
        if len(active_clients) >= MAX_USERS:
            return jsonify({'error': 'Server at maximum capacity. Please try again later.'}), 503
    
    audio_file = request.files['audio']
    
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        audio_file.save(temp_file.name)
        temp_filename = temp_file.name
    
    try:
        print(f"[Flask API] Processing full audio file: {temp_filename}")
        
        # Transcribe the audio file with model lock to prevent GPU memory issues
        with model_lock:
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

@app.route('/status', methods=['GET'])
def server_status():
    with client_lock:
        active_count = len(active_clients)
        client_list = [{'id': cid[:8] + '...', 'address': data['address'], 'connected_since': data['last_activity']} 
                      for cid, data in active_clients.items()]
    
    return jsonify({
        'status': 'online',
        'active_clients': active_count,
        'max_clients': MAX_USERS,
        'available_slots': MAX_USERS - active_count,
        'uptime': time.time() - server_start_time,
        'clients': client_list
    })

def run_flask_app():
    print(f"Starting Flask API server on {HOST}:{FLASK_PORT}...")
    app.run(host=HOST, port=FLASK_PORT, debug=False, use_reloader=False, threaded=True)

#===================#
# Main Entry Point
#===================#

if __name__ == "__main__":
    print("Starting Multi-User Speech Transcription System...")
    server_start_time = time.time()
    
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask_app)
    flask_thread.daemon = True  # This ensures the thread will exit when the main program exits
    flask_thread.start()
    
    # Run the WebSocket server in the main thread
    asyncio.run(run_websocket_server())
