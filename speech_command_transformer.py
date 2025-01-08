import os
os.environ["TORCHAUDIO_BACKEND"] = "sox_io"
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader

# SpeechCommands 데이터셋의 서브셋(훈련/검증/테스트)을 로드하는 클래스
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./data", download=True)
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            if not os.path.exists(filepath):
                return []
            with open(filepath) as f:
                return [os.path.join(self._path, line.strip()) for line in f]
        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

# Transformer 기반 음성 명령 인식 모델
class SpeechTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super(SpeechTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)  # 입력을 임베딩 차원으로 투영
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )
        # 여러 Transformer 인코더 레이어를 쌓음
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)  # 최종 분류기
        
    def forward(self, x):
        x = self.input_projection(x)      # 입력 투영
        x = x.transpose(0, 1)             # Transformer 입력 형태로 변환: (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)   # Transformer 인코더 통과
        x = x.mean(dim=0)                 # 시퀀스 차원 평균
        return self.classifier(x)         # 클래스 로짓 반환

# 테스트 데이터셋에 대한 모델 평가 함수
def evaluate(model, data_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total

def main():
    # Mel-spectrogram 변환기 정의
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
    # 데이터셋 로드
    train_set = SubsetSC("training")
    test_set = SubsetSC("testing")

    # 라벨 집합 구축 및 인덱스 매핑
    labels = sorted(list(set(dat[2] for dat in train_set)))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    # 배치 전처리 함수: Mel-spectrogram 변환 및 시퀀스 패딩
    def collate_fn(batch):
        waveforms, targets = [], []
        for waveform, sample_rate, label, *_ in batch:
            mel_spec = transform(waveform).mean(dim=0).transpose(0, 1)
            waveforms.append(mel_spec)
            targets.append(label_to_idx[label])
        waveforms = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        return waveforms, torch.tensor(targets)

    # DataLoader 생성
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 모델, 손실 함수, 옵티마이저 초기화
    input_dim = 128
    num_classes = len(labels)
    model = SpeechTransformer(input_dim=input_dim, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 학습 루프
    for epoch in range(10):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/10, Loss: {total_loss / len(train_loader):.4f}")

    # 평가
    accuracy = evaluate(model, test_loader)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
