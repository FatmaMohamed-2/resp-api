from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import os, tempfile
import joblib
import librosa
import requests
import tensorflow as tf

# =========================
# Config
# =========================
MODEL_URL = (os.environ.get("MODEL_URL") or "").strip()  # important: strip newline/spaces
MODEL_PATH = "resp_cnn.tflite"  # keep same as your asset name
ENCODER_PATH = "label_encoder.pkl"

TARGET_SR = 22050
SECONDS = 20.0
EXPECTED_FEATURES = 193

app = FastAPI(title="Respiratory Audio CNN API")

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

    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()

    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def load_audio_20s(path: str, target_sr=TARGET_SR, seconds=SECONDS):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    target_len = int(seconds * sr)
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    return y, sr


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
    tonnetz = np.mean(
        librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr).T, axis=0
    )
    feat = np.hstack([mfcc, chroma, mel, contrast, tonnetz]).astype(np.float32)  # 193
    return feat


# =========================
# Startup
# =========================
@app.on_event("startup")
def startup():
    global interpreter, input_details, output_details, label_encoder

    # 1) Download model from GitHub Release
    download_if_needed(MODEL_URL, MODEL_PATH)

    # 2) Load encoder (must be in repo)
    if not os.path.exists(ENCODER_PATH):
        raise RuntimeError(f"Encoder not found: {ENCODER_PATH}")

    label_encoder = joblib.load(ENCODER_PATH)

    # 3) Load TFLite using TensorFlow's interpreter (NOT tflite_runtime)
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # sanity check (optional)
    # expected input shape: (1, 193, 1)
    # print("Input details:", input_details)
    # print("Output details:", output_details)


# =========================
# Routes
# =========================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_url": MODEL_URL,
        "model_path_exists": os.path.exists(MODEL_PATH),
        "encoder_exists": os.path.exists(ENCODER_PATH),
    }


@app.post("/predict_audio")
async def predict_audio(file: UploadFile = File(...)):
    if interpreter is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    suffix = os.path.splitext(file.filename or "")[1].lower() or ".wav"
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(await file.read())

        y, sr = load_audio_20s(tmp_path)
        feat = extract_features(y, sr)

        if feat.shape[0] != EXPECTED_FEATURES:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {EXPECTED_FEATURES} features, got {feat.shape[0]}",
            )

        x = feat.reshape(1, EXPECTED_FEATURES, 1).astype(np.float32)

        # Run inference
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        probs = interpreter.get_tensor(output_details[0]["index"])[0].astype(np.float32)

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
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
