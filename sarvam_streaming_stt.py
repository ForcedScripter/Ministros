import requests
import tempfile
import os
from config import SARVAM_API_KEY

# ==================================
# CONFIG
# ==================================

STT_URL = "https://api.sarvam.ai/speech-to-text"

RATE = 16000
CHANNELS = 1
CHUNK_SECONDS = 3


# ==================================
# TRANSCRIBE
# ==================================


def transcribe_chunk(audio_path):
    url = "https://api.sarvam.ai/speech-to-text"

    headers = {"Authorization": f"Bearer {SARVAM_API_KEY}"}

    files = {"file": ("audio.wav", open(audio_path, "rb"), "audio/wav")}

    data = {"model": "saarika:v2.5"}

    response = requests.post(url, headers=headers, files=files, data=data, timeout=30)

    print("STATUS:", response.status_code)
    print("RAW RESPONSE:", response.text)

    if response.status_code != 200:
        return ""

    result = response.json()

    return result.get("transcript", "")


# ==================================
# STREAM LOOP (local CLI only — requires sounddevice + PortAudio)
# ==================================


def stream_audio_to_sarvam(callback):
    # Lazy imports — these require PortAudio system library which
    # is NOT available on Render / cloud servers.  Only used for
    # local microphone streaming, never by the FastAPI server.
    import sounddevice as sd
    import soundfile as sf

    print("\n🎤 Speak continuously (Ctrl+C to stop)\n")

    try:
        while True:
            print("Recording chunk...")

            audio = sd.rec(
                int(CHUNK_SECONDS * RATE),
                samplerate=RATE,
                channels=CHANNELS,
                dtype="int16",
            )

            sd.wait()

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, RATE)

                text = transcribe_chunk(f.name)

            os.remove(f.name)

            if text.strip():
                print("\n✅ TRANSCRIPT:", text)
                callback(text)

    except KeyboardInterrupt:
        print("\n🛑 Streaming stopped.")
