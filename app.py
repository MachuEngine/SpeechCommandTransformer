import os
import boto3
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

BUCKET_NAME = "machu-stt-model-store-20260401-created" 
S3_FOLDER_PREFIX = "my_model/" # S3에 업로드된 파일의 정확한 이름
LOCAL_DIR = "./my_model"

def download_model_folder_from_s3():
    # 로컬에 my_model 폴더가 없다면 새로 만들고 다운로드 시작
    if not os.path.exists(LOCAL_DIR) or not os.listdir(LOCAL_DIR):
        print(f"모델 파일이 없습니다. S3({BUCKET_NAME})에서 폴더 다운로드를 시작합니다...")
        os.makedirs(LOCAL_DIR, exist_ok=True)
        
        s3_client = boto3.client('s3', region_name='ap-northeast-2')
        # 해당 폴더(Prefix) 안에 있는 모든 파일 목록 가져오기
        objects = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=S3_FOLDER_PREFIX)
        
        if 'Contents' in objects:
            for obj in objects['Contents']:
                s3_key = obj['Key']
                
                # S3의 경로가 폴더 자체(이름이 /로 끝남)면 다운로드 건너뛰기
                if s3_key.endswith('/'): 
                    continue
                
                # 로컬에 저장할 최종 경로 만들기 (예: ./my_model/model.pth)
                local_file_path = os.path.join(".", s3_key)
                
                # 파일이 들어갈 하위 폴더가 로컬에 없다면 미리 생성
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                print(f"다운로드 중: {s3_key}")
                s3_client.download_file(BUCKET_NAME, s3_key, local_file_path)
                
            print("✨ 모델 폴더 다운로드 완벽하게 완료!")
        else:
            print("⚠️ S3에 다운로드할 파일이 없습니다. 버킷 이름과 폴더명을 확인해주세요.")
    else:
        print("✅ 로컬에 이미 모델이 존재합니다.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: Optional[torch.nn.Module] = None

_mel_transform = T.MelSpectrogram(sample_rate=INPUT_SAMPLE_RATE, n_mels=N_MELS)

@app.on_event("startup")
def _startup() -> None:
    global _model

    download_model_folder_from_s3()
    
    try:
        # my_model 폴더에서 바로 불러옵니다.
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