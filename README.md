# 📊 Speech Command Recognition with Transformer

This repository provides an example implementation of a Transformer-based speech command recognition model using PyTorch and torchaudio. It covers the complete pipeline to download, preprocess the Speech Commands dataset, train a Transformer model, and evaluate its performance.

## 🚀 Key Features
- **Dataset Loading**: Implements a custom `SubsetSC` class that extends `torchaudio.datasets.SPEECHCOMMANDS` to selectively load training, validation, or testing subsets.
- **Feature Extraction**: Converts raw audio signals into Mel-spectrograms to use as input features.
- **Model Architecture**: Implements a `SpeechTransformer` model using a Transformer Encoder for speech command classification.
- **Training & Evaluation**: Trains the model for 10 epochs and achieves approximately 80% accuracy on the test set.

## ✅ Requirements
- Python 3.6+
- [PyTorch](https://pytorch.org/)
- [torchaudio](https://pytorch.org/audio/stable/index.html)

## ✅ Installation

### Repository Clone

```bash
git clone https://github.com/MachuEngine/SpeechCommandTransformer.git
```
### Required Packages

Install the required packages:
```bash
pip install -r requirements.txt
```


## 📰 Usage
1. Clone the repository and navigate to the project directory.
2. Run the script to start training and evaluation:
python main.py

The script will automatically download the Speech Commands dataset, start model training, and then evaluate the model on the test set.

## ✨ Code Overview

### 1. Dataset Class: `SubsetSC`
The `SubsetSC` class extends the `SPEECHCOMMANDS` dataset to selectively load subsets of the data:
- **Training Set**: Excludes validation and testing files.
- **Validation/Testing Set**: Loads files based on predefined file lists.

### 2. Model Architecture: `SpeechTransformer`
The Transformer-based model processes Mel-spectrogram features through the following steps:
- **Input Projection**: Projects input features to the desired model dimension.
- **Transformer Encoder**: Stacks multiple encoder layers to learn complex feature representations.
- **Classifier**: Maps the encoded features to class logits for classification.

```py
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
        out = self.input_projection(x) 
        out = out.transpose(0, 1) 
        out = self.transformer_encoder(out) 
        out = out.mean(dim=0) 
        logits = self.classifier(out) 
        return logits
```

### 3. Training Pipeline
- **Data Preprocessing**: Converts audio waveforms to Mel-spectrograms, averages across channels, transposes dimensions, and pads sequences to form batches.
- **Loss Function and Optimizer**: Uses CrossEntropyLoss and Adam optimizer for training.
- **Training Loop**: Runs for 10 epochs, printing the average loss for each epoch.

### 4. Evaluation
The `evaluate` function calculates model accuracy on the test set.

## ⚛️ Training and Evaluation Results

The loss progression over epochs during training:

```plaintext
Epoch 1/10, Loss: 1.9380
Epoch 2/10, Loss: 1.2769
Epoch 3/10, Loss: 1.0876
Epoch 4/10, Loss: 0.9995
Epoch 5/10, Loss: 0.9530
Epoch 6/10, Loss: 0.8923
Epoch 7/10, Loss: 0.8470
Epoch 8/10, Loss: 0.8180
Epoch 9/10, Loss: 0.7867
Epoch 10/10, Loss: 0.7556
```

Final evaluation accuracy on the test set:
```plaintext
Test Accuracy: 80.36%
```

This result demonstrates the effectiveness of the Transformer-based model for speech command recognition on the Speech Commands dataset.
