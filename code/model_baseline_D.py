"""Whisper-based intelligibility prediction model"""

import whisper
import torch
import torch.nn as nn
import config

class WhisperIntelligibilityPredictor(nn.Module):
    def __init__(self, model_name: str = None, model_cache: str = None, freeze_encoder: bool = True):
        super().__init__()
        
        model_name = model_name or config.MODEL_NAME
        model_cache = model_cache or config.MODEL_CACHE
        
        # Load pre-trained Whisper
        print(f"Loading Whisper model: {model_name}")
        if model_cache:
            self.whisper = whisper.load_model(model_name, download_root=model_cache)
        else:
            self.whisper = whisper.load_model(model_name)
        
        # Freeze Whisper encoder weights for faster training
        if freeze_encoder:
            print("Freezing Whisper encoder weights")
            for param in self.whisper.parameters():
                param.requires_grad = False
        
        # Encoder dimension (tiny=384, base=512, small=768)
        encoder_dim = 384 if 'tiny' in model_name else 512
        
        # Regression head to predict intelligibility score [0, 1]
        self.regression_head = nn.Sequential(
            nn.Linear(encoder_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        print(f"Model initialized with {sum(p.numel() for p in self.regression_head.parameters())} trainable parameters")
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: Batch of audio tensors [batch_size, samples]
        Returns:
            predictions: Intelligibility scores [batch_size]
        """
        device = next(self.whisper.parameters()).device
        
        # Convert audio to mel spectrograms
        mel_list = []
        for i in range(audio.shape[0]):
            mel = whisper.log_mel_spectrogram(audio[i].cpu().numpy())
            mel_list.append(mel)
        
        mel = torch.stack(mel_list).to(device)
        
        # Extract encoder features
        encoder_output = self.whisper.encoder(mel)
        
        # Global average pooling over time dimension
        features = encoder_output.mean(dim=1)
        
        # Predict intelligibility
        prediction = self.regression_head(features)
        
        return prediction.squeeze()
