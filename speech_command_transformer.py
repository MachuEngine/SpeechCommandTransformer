import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader

# 1. 데이터셋 클래스 정의
class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
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

# 2. Transformer 기반 음성 명령 인식 모델 정의
class SpeechTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, d_model=128, nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super(SpeechTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)             # -> (batch_size, seq_len, d_model)
        x = x.transpose(0, 1)                    # Transformer 입력 shape: (seq_len, batch_size, d_model)
        x = self.transformer_encoder(x)          # -> (seq_len, batch_size, d_model)
        x = x.mean(dim=0)                        # 시퀀스 차원 평균 -> (batch_size, d_model)
        logits = self.classifier(x)              # -> (batch_size, num_classes)
        return logits

# 3. 평가 함수 정의
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total

# 4. main 함수 정의
def main():
    # 데이터 전처리 및 DataLoader 설정
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)

    # 데이터셋 로드
    train_set = SubsetSC("training")
    test_set = SubsetSC("testing")

    # 라벨 목록 및 인덱스 매핑 생성
    labels = sorted(list(set(dat[2] for dat in train_set)))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    def collate_fn(batch):
        waveforms = []
        targets = []
        for waveform, sample_rate, label, *_ in batch:
            # Mel-Spectrogram 변환
            mel_spec = transform(waveform)  # shape: (channel, n_mels, time)
            mel_spec = mel_spec.mean(dim=0)  # 채널 차원 평균 -> shape: (n_mels, time)
            mel_spec = mel_spec.transpose(0, 1)  # shape: (time, n_mels)
            waveforms.append(mel_spec)
            targets.append(label_to_idx[label])
        # 시퀀스 길이를 패딩하여 동일하게 맞추기
        waveforms = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        targets = torch.tensor(targets)
        return waveforms, targets

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 모델 및 학습 설정
    input_dim = 128  # Mel-spectrogram의 n_mels 값
    num_classes = len(labels)
    model = SpeechTransformer(input_dim=input_dim, num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 학습 루프
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)               # 순전파
            loss = criterion(outputs, targets)    # 손실 계산
            loss.backward()                       # 역전파
            optimizer.step()                      # 가중치 업데이트
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # 모델 평가
    accuracy = evaluate(model, test_loader)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    main()
