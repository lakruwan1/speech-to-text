import asyncio
import websockets
import numpy as np
import soundfile as sf
import io
from faster_whisper import WhisperModel

# Server config
HOST = '0.0.0.0'
PORT = 8765

# Whisper model
model_size = "medium.en"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# Audio params
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16 bits = 2 bytes
CHANNELS = 1
CHUNK_DURATION_MS = 1000  # How often to transcribe (in ms)

# Buffer to store audio data
audio_buffer = b''

async def handle_client(websocket, path):
    print(f"Client connected: {websocket.remote_address}")

    global audio_buffer
    audio_buffer = b''

    try:
        async for message in websocket:
            audio_buffer += message

            if len(audio_buffer) >= SAMPLE_RATE * SAMPLE_WIDTH * CHUNK_DURATION_MS / 1000:
                audio_np = np.frombuffer(audio_buffer, dtype=np.int16)

                with io.BytesIO() as wav_io:
                    sf.write(wav_io, audio_np, SAMPLE_RATE, format='WAV')
                    wav_io.seek(0)

                    segments, _ = model.transcribe(wav_io)
                    transcription = " ".join([segment.text for segment in segments])
                    print("\033[92m" + transcription + "\033[0m")

                audio_buffer = b''

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Client disconnected: {e}")

async def main():
    async with websockets.serve(handle_client, HOST, PORT):
        print(f"Server listening on {HOST}:{PORT} ...")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
