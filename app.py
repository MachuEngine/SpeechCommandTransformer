import os
import tempfile
from typing import Optional

import mlflow
import torch
import torch.nn.functional as F
os.environ["TORCHAUDIO_BACKEND"] = "sox_io"
import torchaudio
import torchaudio.transforms as T
from fastapi import FastAPI, File, HTTPException, UploadFile

# ----------------------------
# Configuration
# ----------------------------
INPUT_SAMPLE_RATE = 16000
N_MELS = 128

SPEECHCOMMANDS_LABELS = sorted(
    {
        "backward", "bed", "bird", "cat", "dog", "down", "eight", "five", 
        "follow", "forward", "four", "go", "happy", "house", "learn", "left", 
        "marvin", "nine", "no", "off", "on", "one", "out", "right", "seven", 
        "sheila", "six", "stop", "three", "tree", "two", "up", "visual", "yes", "zero"
    }
)

app = FastAPI(title="Speech Command Transformer - Serving")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: Optional[torch.nn.Module] = None

_mel_transform = T.MelSpectrogram(sample_rate=INPUT_SAMPLE_RATE, n_mels=N_MELS)


@app.on_event("startup")
def _startup() -> None:
    global _model
    try:
        # 복잡한 경로 탐색 안녕! 직관적으로 my_model 폴더에서 바로 불러옵니다.
        _model = mlflow.pytorch.load_model("./my_model", map_location=device)
        _model.eval()
        _model.to(device)
        print("[startup] Model loaded successfully from ./my_model")
    except Exception as e:
        print(f"[startup] Model load failed: {e}")
        _model = None


def _preprocess_wav(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    if sr != INPUT_SAMPLE_RATE:
        resampler = T.Resample(orig_freq=sr, new_freq=INPUT_SAMPLE_RATE)
        waveform = resampler(waveform)

    waveform = waveform.to(torch.float32)

    mel = _mel_transform(waveform)
    mel = mel.mean(dim=0)
    mel = mel.transpose(0, 1)
    x = mel.unsqueeze(0)

    if hasattr(_model, "pos_encoder") and hasattr(_model.pos_encoder, "pe"):
        max_len = int(_model.pos_encoder.pe.size(1))
        if x.size(1) > max_len:
            x = x[:, :max_len, :]

    return x.to(device)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Check startup logs."
        )

    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Please upload a .wav file.")

    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        waveform, sr = torchaudio.load(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {e}")
    finally:
        try:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

    try:
        x = _preprocess_wav(waveform, sr)
        with torch.no_grad():
            logits = _model(x)
            probs = F.softmax(logits, dim=-1)
            top_idx = int(torch.argmax(probs, dim=-1).item())
            top_prob = float(probs[0, top_idx].item())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    label = SPEECHCOMMANDS_LABELS[top_idx] if top_idx < len(SPEECHCOMMANDS_LABELS) else str(top_idx)

    return {
        "label": label,
        "probability": top_prob,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)