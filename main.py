from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import os, tempfile, subprocess
import joblib
import librosa
import requests

import tflite_runtime.interpreter as tflite


# =========================
# Config
# =========================
MODEL_URL = os.environ.get("MODEL_URL")  # direct download URL for resp_cnn.tflite
MODEL_PATH = "resp_cnn.tflite"
ENCODER_PATH = "label_encoder.pkl"

TARGET_SR = 22050
SECONDS = 20.0
EXPECTED_FEATURES = 193

app = FastAPI(title="Respiratory Audio CNN (TFLite) API")

interpreter = None
input_details = None
output_details = None
label_encoder = None


# =========================
# Helpers
# =========================
def download_if_needed(url: str, path: str):
    if os.path.exists(path):
        return
    if not url:
        raise RuntimeError("MODEL_URL env var not set")

    r = requests.get(url, stream=True, timeout=180)
    r.raise_for_status()
    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

def ensure_20s(y: np.ndarray, sr: int, seconds: float = SECONDS):
    target_len = int(seconds * sr)
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    return y

def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0)
    feat = np.hstack([mfcc, chroma, mel, contrast, tonnetz]).astype(np.float32)  # 193
    return feat

def convert_to_wav_22050_mono(input_path: str, output_path: str):
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ac", "1", "-ar", str(TARGET_SR), output_path]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg conversion failed: " + p.stderr.decode(errors="ignore"))


# =========================
# Startup
# =========================
@app.on_event("startup")
def startup():
    global interpreter, input_details, output_details, label_encoder

    # 1) download tflite
    download_if_needed(MODEL_URL, MODEL_PATH)

    # 2) load encoder (must exist in repo)
    if not os.path.exists(ENCODER_PATH):
        raise RuntimeError(f"Encoder not found: {ENCODER_PATH}")
    label_encoder = joblib.load(ENCODER_PATH)

    # 3) init TFLite
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # sanity print (logs)
    print("TFLite input:", input_details)
    print("TFLite output:", output_details)


# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}

@app.post("/predict_audio")
async def predict_audio(file: UploadFile = File(...)):
    if interpreter is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    suffix = os.path.splitext(file.filename or "")[1].lower() or ".bin"
    tmp_in = None
    tmp_wav = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            tmp_in = f.name
            f.write(await file.read())

        tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        convert_to_wav_22050_mono(tmp_in, tmp_wav)

        y, sr = librosa.load(tmp_wav, sr=TARGET_SR, mono=True)
        y = ensure_20s(y, sr)

        feat = extract_features(y, sr)
        if feat.shape[0] != EXPECTED_FEATURES:
            raise HTTPException(status_code=400, detail=f"Expected 193 features, got {feat.shape[0]}")

        x = feat.reshape(1, 193, 1).astype(np.float32)

        # set input + invoke
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details[0]["index"])[0].astype(np.float32)

        pred_idx = int(np.argmax(probs))
        label = str(label_encoder.inverse_transform([pred_idx])[0])
        confidence = float(probs[pred_idx])

        prob_dict = {}
        for i, name in enumerate(list(label_encoder.classes_)):
            if i < len(probs):
                prob_dict[str(name)] = float(probs[i])

        return {"label": label, "confidence": confidence, "probabilities": prob_dict}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    finally:
        for p in [tmp_in, tmp_wav]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass










