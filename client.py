import asyncio
import websockets
import pyaudio

SERVER_URI = "ws://38.65.239.30:35489"  # Updated to match your server port

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

async def send_audio(websocket):
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording and streaming... Press Ctrl+C to stop.")
    try:
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            await websocket.send(data)
    except asyncio.CancelledError:
        print("Audio sending stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

async def receive_transcription(websocket):
    try:
        async for message in websocket:
            print(f"\033[94mTranscription: {message}\033[0m")  # Blue colored transcription output
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Server closed connection: {e}")

async def main():
    async with websockets.connect(SERVER_URI) as websocket:
        send_task = asyncio.create_task(send_audio(websocket))
        receive_task = asyncio.create_task(receive_transcription(websocket))

        # Wait for either task to complete (or for Ctrl+C)
        done, pending = await asyncio.wait(
            [send_task, receive_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Client stopped.")
