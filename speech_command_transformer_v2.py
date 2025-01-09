import os
os.environ["TORCHAUDIO_BACKEND"] = "sox_io"
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torchaudio.datasets import SPEECHCOMMANDS
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
import random

# 위치 인코딩 구현
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

# SpeechCommands 데이터셋의 서브셋 클래스
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
    def __init__(self, input_dim, num_classes, d_model=128, nhead=8, num_layers=4, 
                 dim_feedforward=512, dropout=0.1, max_len=5000):
        super(SpeechTransformer, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.mean(dim=0)
        return self.classifier(x)

# 모델 평가 함수
def evaluate_model(model, data_loader):
    model.eval()
    correct, total = 0, 0
    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    accuracy = correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='macro', zero_division=0
    )
    return accuracy, precision, recall, f1

# 학습 및 평가 실행 함수
def run_training_and_evaluation(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs):
    train_losses = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1s = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        accuracy, precision, recall, f1 = evaluate_model(model, test_loader)
        test_accuracies.append(accuracy)
        test_precisions.append(precision)
        test_recalls.append(recall)
        test_f1s.append(f1)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
              f"Accuracy: {accuracy*100:.2f}%, Precision: {precision*100:.2f}%, "
              f"Recall: {recall*100:.2f}%, F1-score: {f1*100:.2f}%")
    return train_losses, test_accuracies, test_precisions, test_recalls, test_f1s

# 샘플 데이터 시각화 함수
def visualize_samples(dataset, transform, num_samples=5):
    indices = random.sample(range(len(dataset)), num_samples)
    fig, axs = plt.subplots(num_samples, 2, figsize=(12, 3 * num_samples))
    for i, idx in enumerate(indices):
        waveform, sr, label, *_ = dataset[idx]
        mel_spec = transform(waveform).mean(dim=0)  # 채널 평균
        
        # 파형 시각화
        axs[i, 0].plot(waveform.t().numpy())
        axs[i, 0].set_title(f"Waveform - Label: {label}")
        axs[i, 0].set_xlabel("Time")
        axs[i, 0].set_ylabel("Amplitude")
        
        # Mel-spectrogram 시각화
        im = axs[i, 1].imshow(mel_spec.numpy(), origin='lower', aspect='auto', cmap='viridis')
        axs[i, 1].set_title(f"Mel-Spectrogram - Label: {label}")
        axs[i, 1].set_xlabel("Time")
        axs[i, 1].set_ylabel("Mel Frequency Bin")
        fig.colorbar(im, ax=axs[i, 1])
    plt.tight_layout()
    plt.show()

def main():
    transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128)
    train_set = SubsetSC("training")
    test_set = SubsetSC("testing")

    # 데이터 시각화: 5개 샘플의 파형, Mel-spectrogram, 라벨 표시
    visualize_samples(train_set, transform, num_samples=5)

    labels = sorted(list(set(dat[2] for dat in train_set)))
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    def collate_fn(batch):
        waveforms, targets = [], []
        for waveform, sample_rate, label, *_ in batch:
            mel_spec = transform(waveform).mean(dim=0).transpose(0, 1)
            waveforms.append(mel_spec)
            targets.append(label_to_idx[label])
        waveforms = nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        return waveforms, torch.tensor(targets)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, collate_fn=collate_fn)

    input_dim = 128
    num_classes = len(labels)
    model = SpeechTransformer(input_dim=input_dim, num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    epochs = 10
    train_losses, test_accuracies, test_precisions, test_recalls, test_f1s = run_training_and_evaluation(
        model, train_loader, test_loader, criterion, optimizer, scheduler, epochs
    )

    # 시각화: 학습 손실, 정확도, Precision, Recall, F1-score 그래프
    epochs_range = range(1, epochs + 1)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # 첫 번째 서브플롯: Loss & Accuracy
    ax1 = axs[0]
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs_range, train_losses, color=color, label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    ax1b = ax1.twinx()
    color = 'tab:red'
    ax1b.set_ylabel('Accuracy', color=color)
    ax1b.plot(epochs_range, test_accuracies, color=color, label='Test Accuracy')
    ax1b.tick_params(axis='y', labelcolor=color)
    ax1b.legend(loc='upper right')

    # 두 번째 서브플롯: Precision, Recall, F1-score
    ax2 = axs[1]
    ax2.plot(epochs_range, test_precisions, label='Precision')
    ax2.plot(epochs_range, test_recalls, label='Recall')
    ax2.plot(epochs_range, test_f1s, label='F1-score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.set_title('Precision, Recall, and F1-score over Epochs')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
