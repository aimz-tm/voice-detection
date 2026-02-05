from fastapi import FastAPI, HTTPException, UploadFile, File
from pydub import AudioSegment
import io
import numpy as np
from pydub.utils import which
import json

# Tell pydub where ffmpeg is (your path)
AudioSegment.converter = r"D:\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"D:\ffmpeg-8.0.1-essentials_build\ffmpeg-8.0.1-essentials_build\bin\ffprobe.exe"

from model import detect_voice

app = FastAPI(
    title="AI Voice Detector",
    description="Detects AI-generated voices with multi-language support"
)

@app.post("/detect-voice")
async def detect_voice_endpoint(file: UploadFile = File(...)):
    try:
        # Read uploaded file
        audio_bytes = await file.read()

        # Let pydub auto-detect format (wav/mp3/m4a)
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=file.filename.split(".")[-1])

        # Convert to mono
        audio = audio.set_channels(1)

        # Convert to numpy
        audio_array = np.array(audio.get_array_of_samples(), dtype=np.float32)

        # Normalize
        audio_array /= 32768.0

        sr = audio.frame_rate

        classification, confidence, explanation, detected_language = detect_voice(
            audio_array, sr
        )

        return {
            "classification": classification,
            "confidence": round(confidence, 4),
            "explanation": explanation,
            "detected_language": detected_language
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing audio: {str(e)}"
        )

@app.get("/")
async def root():
    return {"message": "AI Voice Detection API is running"}

