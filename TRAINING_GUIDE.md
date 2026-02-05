# AI Voice Detection - Training & Deployment Guide

## Overview
This system detects whether a voice sample is AI-generated or human-generated across multiple languages (Tamil, English, Hindi, Malayalam, Telugu).

## Key Improvements
- **Better Training Data**: Uses Kaggle dataset instead of random data
- **Improved Model**: RandomForestClassifier with optimized hyperparameters (200 estimators, better tree depth)
- **Feature Extraction**: Wav2Vec2 embeddings + Mel spectrogram analysis
- **Model Persistence**: Trained model saved to disk for faster API startup

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Kaggle API (Optional but Recommended)
To use the real Kaggle dataset for training:
1. Download API credentials from https://kaggle.com/settings/account (kaggle.json)
2. Place in `~/.kaggle/kaggle.json` on your system
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac)

### 3. Train the Model
```bash
python train.py
```

**What happens:**
- Attempts to load the Kaggle dataset: "speech-dataset-of-human-and-ai-generated-voices"
- Extracts Wav2Vec2 features + Mel spectrogram from each audio file
- Trains RandomForestClassifier with 200 estimators
- Saves trained model to `voice_detector_model.pkl`
- Falls back to synthetic training data if Kaggle download fails

**Expected output:**
```
============================================================
KAGGLE DATASET LOADER
============================================================

Loading Kaggle dataset...
Dataset loaded! Shape: (XXXXX, YY)
...
============================================================
TRAINING CLASSIFIER
============================================================

Training on XXXX samples...
Training accuracy: 0.XXXX
Model saved to: voice_detector_model.pkl
```

### 4. Run the API
```bash
uvicorn main:app --reload
```

The API will:
- Automatically load the trained model from `voice_detector_model.pkl`
- Start on `http://localhost:8000`
- Provide interactive documentation at `http://localhost:8000/docs`

## API Endpoints

### POST /detect-voice
Upload an audio file to detect if it's AI-generated or human-generated.

**Request:**
```
multipart/form-data:
  file: <audio_file> (MP3, WAV, M4A, etc.)
```

**Response:**
```json
{
  "classification": "AI-generated" | "Human-generated",
  "confidence": 0.0-1.0,
  "explanation": "Detailed explanation of the classification",
  "detected_language": "English" | "Tamil" | "Hindi" | "Malayalam" | "Telugu" | "unknown"
}
```

### GET /
Health check endpoint.

## Model Architecture

### Feature Extraction
1. **Wav2Vec2 Embeddings** (768 features)
   - Pre-trained on LibriSpeech
   - Captures linguistic and acoustic patterns
   
2. **Mel Spectrogram** (128 features)
   - Standard audio feature for speech analysis
   - Captures frequency patterns

3. **Combined Features** (896 total dimensions)

### Classification
- **Algorithm**: Random Forest (200 trees)
- **Labels**: 0 = Human, 1 = AI-generated
- **Output**: Probability distribution for confidence scoring

## Language Detection
- Uses Whisper (speech-to-text) to extract text from audio
- Uses langid library to identify language
- Supports: Tamil, English, Hindi, Malayalam, Telugu

## Improving Model Performance

### If confidence is still low:
1. **Check training data quality**: Look at Kaggle dataset for audio variety
2. **Adjust hyperparameters** in `train.py`:
   ```python
   classifier = RandomForestClassifier(
       n_estimators=300,      # Increase for more complex patterns
       max_depth=30,          # Deeper trees
       min_samples_split=3,   # More aggressive splitting
       min_samples_leaf=1,
       random_state=42,
       n_jobs=-1
   )
   ```

3. **Use more training samples**: 
   - Collect additional voice samples
   - Add to the training pipeline

4. **Feature engineering**:
   - Add MFCC (Mel-frequency cepstral coefficients)
   - Add zero-crossing rate
   - Add spectral centroid

## Troubleshooting

### "No trained model found" message
- This is normal on first run. The model will be created with synthetic data.
- Run `train.py` to train on real Kaggle data.

### Kaggle dataset download fails
- Ensure `~/.kaggle/kaggle.json` exists with correct permissions
- Or manually download from: https://kaggle.com/datasets/kambingbersayaphitam/speech-dataset-of-human-and-ai-generated-voices
- Place files in a directory and modify `train.py` to read locally

### FFmpeg errors
- Update the ffmpeg path in `main.py` if your installation is different:
  ```python
  AudioSegment.converter = r"C:\path\to\ffmpeg.exe"
  AudioSegment.ffprobe = r"C:\path\to\ffprobe.exe"
  ```

## Files
- `train.py` - Model training script
- `model.py` - Feature extraction & prediction logic
- `main.py` - FastAPI server
- `requirements.txt` - Python dependencies
- `voice_detector_model.pkl` - Saved trained model (created after training)

## Next Steps
1. Run `train.py` to train on Kaggle dataset
2. Run `uvicorn main:app` to start the API
3. Test with `/docs` endpoint in browser
4. Monitor confidence scores and retrain if needed
