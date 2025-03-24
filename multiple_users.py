import asyncio
import websockets
import numpy as np
import soundfile as sf
import io
from faster_whisper import WhisperModel

HOST = '0.0.0.0'
PORT = 8765

model_size = "medium.en"
# model_size = "large"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1
CHUNK_DURATION_MS = 1000

async def handle_client(websocket):
    print(f"Client connected: {websocket.remote_address}")

    audio_buffer = b''

    try:
        async for message in websocket:
            audio_buffer += message

            if len(audio_buffer) >= SAMPLE_RATE * SAMPLE_WIDTH * CHUNK_DURATION_MS / 1000:
                audio_np = np.frombuffer(audio_buffer, dtype=np.int16)

                with io.BytesIO() as wav_io:
                    sf.write(wav_io, audio_np, SAMPLE_RATE, format='WAV')
                    wav_io.seek(0)

                    segments, _ = await asyncio.to_thread(model.transcribe, wav_io)
                    transcription = " ".join([segment.text for segment in segments])

                    print(f"[{websocket.remote_address}] {transcription}")

                    # Send back to client
                    await websocket.send(transcription)

                audio_buffer = b''

    except websockets.exceptions.ConnectionClosed as e:
        print(f"Client disconnected: {e}")

async def main():
    async with websockets.serve(
        handle_client,
        HOST,
        PORT,
        ping_interval=60,
        ping_timeout=60
    ):
        print(f"Server listening on {HOST}:{PORT} ...")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
