import os
import gdown

MODEL_ID = os.environ.get("MODEL_ID")
OUT = "cnn_respiratory_model.h5"

if not os.path.exists(OUT):
    if not MODEL_ID:
        raise RuntimeError("MODEL_ID env var not set")
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, OUT, quiet=False)
else:
    print("Model already exists")
