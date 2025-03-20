from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import pyaudio
import wave
import os
from faster_whisper import WhisperModel
import threading

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Define recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
chunk_length = 1  # in seconds

# Initialize Whisper Model
model_size = "medium.en"
# model = WhisperModel(model_size, device="cpu", compute_type="float32")
model = WhisperModel(model_size, device="cuda", compute_type="float16")

is_recording = False
p = pyaudio.PyAudio()
stream = None

def record_and_transcribe():
    global is_recording, stream
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=4096)
    
    while is_recording:
        chunk_file = "temp_chunk.wav"
        frames = []
        
        for _ in range(0, int(RATE / CHUNK * chunk_length)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        
        wf = wave.open(chunk_file, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        transcription = transcribe_chunk(chunk_file)
        os.remove(chunk_file)

        socketio.emit('transcription', {'text': transcription})

def transcribe_chunk(file_path):
    segments, _ = model.transcribe(file_path)
    return " ".join(segment.text for segment in segments)

@app.route('/')
def index():
    return render_template('index12.html')

@socketio.on('start_listening')
def start_listening():
    global is_recording
    if not is_recording:
        is_recording = True
        threading.Thread(target=record_and_transcribe, daemon=True).start()

@socketio.on('stop_listening')
def stop_listening():
    global is_recording, stream
    is_recording = False
    if stream:
        stream.stop_stream()
        stream.close()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
