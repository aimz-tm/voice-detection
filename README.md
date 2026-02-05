# AI Voice Detection

A FastAPI service that classifies short voice samples as AI-generated or Human-generated. The project supports multi-language detection (Tamil, English, Hindi, Malayalam, Telugu) and uses audio features (mel spectrogram, MFCC, spectral features) with a RandomForest classifier.

## Repository Structure
- `main.py` — FastAPI server and endpoints
- `model.py` — feature extraction, language detection, and prediction logic
- `train.py` — training script (uses Kaggle dataset if available, otherwise synthetic data)
- `requirements.txt` — Python dependencies
- `Dockerfile` — production Docker image
- `voice_detector_model.pkl` — trained classifier (created after running `train.py`)
- `TRAINING_GUIDE.md`, `DEPLOYMENT.md`, `RAILWAY_DEPLOY.md` — guides

## Quick Start (Local)
1. Create a Python environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate    # macOS/Linux
.venv\Scripts\activate      # Windows PowerShell
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) If you will use Whisper or PyTorch features, install appropriate `torch` wheel for your platform.

4. Run the API locally:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` to see the interactive API docs.

## Training the Model
To train a new model (recommended if you want to improve accuracy):

1. (Optional) Configure Kaggle credentials at `~/.kaggle/kaggle.json` if you want to use the Kaggle dataset.
2. Run training:

```bash
python train.py
```

This will attempt to download the Kaggle dataset `kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices`. If it cannot, it will fall back to generating synthetic training data. After training the classifier will be saved to `voice_detector_model.pkl`.

Notes:
- If you generate the model locally, the `scikit-learn` version used for training should match the one used at runtime. We pinned `scikit-learn==1.5.1` in `requirements.txt` to match the training environment.
- Training can be slow when extracting features; consider running on a machine with a GPU for heavy feature extraction.

## API Endpoints
- `GET /` — Health check
- `POST /detect-voice` — Multipart form upload. Field name: `file` (MP3/WAV/M4A). Returns JSON:

```json
{
  "classification": "AI-generated" | "Human-generated",
  "confidence": 0.0-1.0,
  "explanation": "...",
  "detected_language": "English" | "Tamil" | "Hindi" | "Malayalam" | "Telugu" | "unknown"
}
```

Example curl:

```bash
curl -X POST "http://localhost:8000/detect-voice" -F "file=@/path/to/sample.mp3"
```

## Deployment (Railway)
We recommend Railway or Render for this project because FastAPI audio processing can exceed serverless time limits.

Quick steps:

1. Push repo to GitHub.
2. Create a Railway project and select "Deploy from GitHub".
3. Railway will auto-detect the `Dockerfile` and build the container.
4. After deployment, open `https://<your-project>.railway.app/docs`.

Make sure `voice_detector_model.pkl` is present in the repo or that `train.py` is run during startup to generate the model.

## Troubleshooting
- `RuntimeError: Form data requires "python-multipart"` — ensure `python-multipart` is in `requirements.txt` (already added).
- `InconsistentVersionWarning` during unpickle — pin `scikit-learn` to the same version used to create the pickle. We pinned `scikit-learn==1.5.1`.
- FFmpeg: `pydub` requires `ffmpeg` installed on the host. For local Windows, set `AudioSegment.converter` and `ffprobe` paths in `main.py` or install `ffmpeg` on the server.

## Notes & Next Steps
- For production stability consider exporting the classifier to ONNX or retraining in the environment matching your deployed runtime.
- If memory or cold-starts are an issue, offload heavy tasks (Whisper transcription, Wav2Vec features) to a separate service or precompute features.

## Contact
If you want, I can:
- Tail Railway logs and fix remaining issues
- Convert the model to ONNX and update `model.py`
- Add CI to run tests and auto-deploy

