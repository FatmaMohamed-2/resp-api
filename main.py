from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import os, tempfile
import joblib
from tensorflow.keras.models import load_model
import librosa
import gdown

MODEL_ID = os.environ.get("MODEL_ID")

def download_model_if_needed():
    if not os.path.exists(MODEL_PATH):
        if not MODEL_ID:
            raise RuntimeError("MODEL_ID env var not set")
        url = f"https://drive.google.com/uc?id={MODEL_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)


app = FastAPI(title="Respiratory Audio CNN API")

MODEL_PATH = "cnn_respiratory_model.h5"
ENCODER_PATH = "label_encoder.pkl"

model = None
label_encoder = None

def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    feat = np.hstack([mfcc, chroma, mel, contrast, tonnetz]).astype(np.float32)  # 193
    return feat

def load_audio_20s(path: str, target_sr=22050, seconds=20.0):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    target_len = int(seconds * sr)
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    return y, sr

@app.on_event("startup")
def startup():
    global model, label_encoder

    download_model_if_needed()

    if not os.path.exists(ENCODER_PATH):
        raise RuntimeError(f"Encoder not found: {ENCODER_PATH}")

    model = load_model(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_audio")
async def predict_audio(file: UploadFile = File(...)):
    if model is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    suffix = os.path.splitext(file.filename or "")[1].lower() or ".wav"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())

        y, sr = load_audio_20s(tmp_path, target_sr=22050, seconds=20.0)
        feat = extract_features(y, sr)
        if feat.shape[0] != 193:
            raise HTTPException(status_code=400, detail=f"Expected 193 features, got {feat.shape[0]}")

        x = feat.reshape(1, 193, 1)
        probs = np.array(model.predict(x, verbose=0)[0], dtype=np.float32)

        pred_idx = int(np.argmax(probs))
        label = str(label_encoder.inverse_transform([pred_idx])[0])
        confidence = float(probs[pred_idx])

        prob_dict = {}
        try:
            for i, name in enumerate(list(label_encoder.classes_)):
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
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass




