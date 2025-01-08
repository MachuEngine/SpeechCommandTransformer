import torch.nn as nn

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