from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import os, tempfile, subprocess
import joblib
from tensorflow.keras.models import load_model
import librosa
import requests

# =========================
# Config (ENV)
# =========================
MODEL_URL = os.getenv("MODEL_URL")  # direct download URL (GitHub Release asset)
ENCODER_URL = os.getenv("ENCODER_URL")  # optional: direct url for encoder (or keep in repo)

MODEL_PATH = os.getenv("MODEL_PATH", "cnn_respiratory_model.h5")
ENCODER_PATH = os.getenv("ENCODER_PATH", "label_encoder.pkl")

TARGET_SR = int(os.getenv("TARGET_SR", "22050"))
SECONDS = float(os.getenv("SECONDS", "20.0"))
EXPECTED_FEATURES = int(os.getenv("EXPECTED_FEATURES", "193"))

# =========================
# App
# =========================
app = FastAPI(title="Respiratory Audio CNN API")

model = None
label_encoder = None


# =========================
# Download helpers
# =========================
def _download_file(url: str, out_path: str, timeout: int = 180):
    if not url:
        raise RuntimeError("Download URL not set")

    # simple retry
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return
        except Exception as e:
            if attempt == 2:
                raise RuntimeError(f"Failed downloading {url}: {e}")

def download_model_if_needed():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 5_000_000:
        return
    if not MODEL_URL:
        raise RuntimeError("MODEL_URL env var not set")
    _download_file(MODEL_URL, MODEL_PATH)

def download_encoder_if_needed():
    if os.path.exists(ENCODER_PATH) and os.path.getsize(ENCODER_PATH) > 10:
        return
    # If encoder is in repo, no need for URL
    if ENCODER_URL:
        _download_file(ENCODER_URL, ENCODER_PATH)
    if not os.path.exists(ENCODER_PATH):
        raise RuntimeError(f"Encoder not found: {ENCODER_PATH} (upload it or set ENCODER_URL)")


# =========================
# Audio helpers
# =========================
def convert_to_wav_22050_mono(input_path: str, output_path: str):
    """
    Convert ANY audio type (m4a/aac/3gp/mp3/...) -> wav mono 22050 using ffmpeg.
    """
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", str(TARGET_SR), output_path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg conversion failed: " + p.stderr.decode(errors="ignore"))

def load_audio_20s_any_format(path: str):
    """
    Always convert to wav first then load with librosa.
    Guarantees consistent SR/mono for mobile recordings.
    """
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    try:
        convert_to_wav_22050_mono(path, tmp_wav)
        y, sr = librosa.load(tmp_wav, sr=TARGET_SR, mono=True)
        target_len = int(SECONDS * sr)
        if len(y) > target_len:
            y = y[:target_len]
        elif len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode="constant")
        return y, sr
    finally:
        try:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        except:
            pass

def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)

    feat = np.hstack([mfcc, chroma, mel, contrast, tonnetz]).astype(np.float32)  # 193
    return feat


# =========================
# Startup
# =========================
@app.on_event("startup")
def startup():
    global model, label_encoder

    # 1) Download assets if needed
    download_model_if_needed()
    download_encoder_if_needed()

    # 2) Load model + encoder
    # IMPORTANT: compile=False
    try:
        model = load_model(MODEL_PATH, compile=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

    try:
        label_encoder = joblib.load(ENCODER_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load encoder from {ENCODER_PATH}: {e}")


# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_exists": os.path.exists(MODEL_PATH),
        "encoder_exists": os.path.exists(ENCODER_PATH),
        "target_sr": TARGET_SR,
        "seconds": SECONDS,
        "expected_features": EXPECTED_FEATURES
    }

@app.post("/predict_audio")
async def predict_audio(file: UploadFile = File(...)):
    if model is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    suffix = os.path.splitext(file.filename or "")[1].lower() or ".bin"
    tmp_in = None

    try:
        # Save upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_in = tmp.name
            tmp.write(await file.read())

        # Load audio safely
        y, sr = load_audio_20s_any_format(tmp_in)

        # Features
        feat = extract_features(y, sr)
        if feat.shape[0] != EXPECTED_FEATURES:
            raise HTTPException(status_code=400, detail=f"Expected {EXPECTED_FEATURES} features, got {feat.shape[0]}")

        # Model input
        x = feat.reshape(1, EXPECTED_FEATURES, 1)
        probs = np.array(model.predict(x, verbose=0)[0], dtype=np.float32)

        pred_idx = int(np.argmax(probs))
        label = str(label_encoder.inverse_transform([pred_idx])[0])
        confidence = float(probs[pred_idx])

        prob_dict = {}
        try:
            for i, name in enumerate(list(label_encoder.classes_)):
                if i < len(probs):
                    prob_dict[str(name)] = float(probs[i])
        except Exception:
            for i in range(len(probs)):
                prob_dict[f"class_{i}"] = float(probs[i])

        return {"label": label, "confidence": confidence, "probabilities": prob_dict}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    finally:
        if tmp_in and os.path.exists(tmp_in):
            try:
                os.remove(tmp_in)
            except:
                pass








