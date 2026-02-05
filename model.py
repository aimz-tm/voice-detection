import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import langid
import pickle
import os

try:
    import whisper
    whisper_model = whisper.load_model("tiny")
    WHISPER_AVAILABLE = True
except Exception as e:
    print(f"Warning: Whisper not available: {e}")
    WHISPER_AVAILABLE = False
    whisper_model = None

FEATURE_DIM = 128 + 13 + 13 + 2  # mels + mfcc + stft + statistical features = 156

# Load trained classifier or create a new one
def load_or_create_classifier():
    model_path = "voice_detector_model.pkl"
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}...")
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        print("No trained model found. Creating default classifier...")
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        # Still need to fit with some data
        X_train = np.random.rand(100, FEATURE_DIM)
        y_train = np.random.randint(0, 2, 100)
        clf.fit(X_train, y_train)
        return clf

classifier = load_or_create_classifier()

def extract_features(audio_array, sr):
    """Extract features from audio using librosa"""
    try:
        # Normalize
        audio_array = audio_array.astype(np.float32)
        audio_array = audio_array / (np.max(np.abs(audio_array)) + 1e-9)
        
        # Resample to 16kHz
        audio = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

        # Mel spectrogram features (128)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128)
        mel_mean = np.mean(mel_spec, axis=1)

        # MFCC features (13)
        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Spectral centroid and rolloff
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=16000)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=16000)
        
        spec_features = np.concatenate([
            np.mean(spectral_centroid),
            np.mean(spectral_rolloff)
        ])

        # Combine all features
        combined = np.concatenate([mel_mean, mfcc_mean, spec_features])
        
        # Ensure fixed size
        if combined.shape[0] < FEATURE_DIM:
            combined = np.pad(combined, (0, FEATURE_DIM - combined.shape[0]))
        else:
            combined = combined[:FEATURE_DIM]
        
        return combined.astype(np.float32)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros(FEATURE_DIM, dtype=np.float32)

def detect_voice(audio_array, sr):
    features = extract_features(audio_array, sr)

    prob = classifier.predict_proba([features])[0]
    confidence = float(max(prob))
    classification = "AI-generated" if prob[1] > 0.5 else "Human-generated"

    explanation = (
        "Uniform spectral patterns indicate AI-generated speech."
        if classification == "AI-generated"
        else
        "Natural prosody and spectral variation suggest human speech."
    )

    detected_language = detect_language_from_audio(audio_array, sr)

    return classification, confidence, explanation, detected_language


def detect_language_from_audio(audio_array, sr):
    """
    Detect language from speech using Whisper + langid
    """
    if not WHISPER_AVAILABLE or whisper_model is None:
        return "unknown"
    
    try:
        # Whisper expects float32 numpy at 16kHz
        if sr != 16000:
            audio_array = librosa.resample(y=audio_array, orig_sr=sr, target_sr=16000)

        # Transcribe (short, fast)
        result = whisper_model.transcribe(
            audio_array,
            language=None,
            task="transcribe",
            fp16=False
        )

        text = result.get("text", "").strip()

        if not text:
            return "unknown"

        lang, _ = langid.classify(text)

        supported = {
            "en": "English",
            "ta": "Tamil",
            "hi": "Hindi",
            "ml": "Malayalam",
            "te": "Telugu"
        }

        return supported.get(lang, "unknown")
    except Exception as e:
        print(f"Error detecting language: {e}")
        return "unknown"
