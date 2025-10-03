"""
Prosody Models Module
- Implements hierarchical prosody prediction models
- Handles duration, pitch, energy, and stress patterns
- Supports emotion-based prosody modification
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np

class DurationPredictor(nn.Module):
    """Predicts phoneme durations"""
    def __init__(self, input_size: int = 768, hidden_size: int = 256):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1)
        )
        self.lstm = nn.LSTM(hidden_size, hidden_size // 2, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, channels)
        x = x.transpose(1, 2)  # (batch, channels, time)
        x = self.conv_layers(x)
        x = x.transpose(1, 2)  # (batch, time, channels)
        x, _ = self.lstm(x)
        return self.proj(x).squeeze(-1)  # (batch, time)

class PitchPredictor(nn.Module):
    """Predicts F0 contours with emotion influence"""
    def __init__(self, input_size: int = 768, hidden_size: int = 256):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, hidden_size),
            nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.GroupNorm(8, hidden_size)
        )
        self.lstm = nn.LSTM(hidden_size, hidden_size//2, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor, emotion_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x.transpose(1, 2)
        conv_out = self.conv_net(x)
        
        # Include emotion influence if provided
        if emotion_embedding is not None:
            # Broadcast emotion embedding across time
            emotion_embedding = emotion_embedding.unsqueeze(-1).expand(-1, -1, conv_out.size(-1))
            conv_out = conv_out + emotion_embedding
            
        lstm_in = conv_out.transpose(1, 2)
        lstm_out, _ = self.lstm(lstm_in)
        return self.proj(lstm_out).squeeze(-1)

class EnergyPredictor(nn.Module):
    """Predicts energy/intensity contours"""
    def __init__(self, input_size: int = 768, hidden_size: int = 256):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.LayerNorm([hidden_size]),
            nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
            nn.ReLU(),
            nn.LayerNorm([hidden_size])
        )
        self.gru = nn.GRU(hidden_size, hidden_size//2, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv_net(x)
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        return self.proj(x).squeeze(-1)

class EmotionalProsodyAdapter(nn.Module):
    """Adapts prosody based on emotional context"""
    def __init__(self, num_emotions: int = 8, hidden_size: int = 256):
        super().__init__()
        self.emotion_embedding = nn.Embedding(num_emotions, hidden_size)
        self.emotion_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Prosody modification networks
        self.pitch_mod = nn.Linear(hidden_size, hidden_size)
        self.energy_mod = nn.Linear(hidden_size, hidden_size)
        self.duration_mod = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, 
                emotion_id: torch.Tensor,
                pitch: torch.Tensor,
                energy: torch.Tensor,
                duration: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Modify prosody features based on emotion
        
        Args:
            emotion_id: Tensor of emotion indices
            pitch: Original pitch predictions
            energy: Original energy predictions
            duration: Original duration predictions
            
        Returns:
            Modified pitch, energy, and duration tensors
        """
        # Get emotion embeddings
        emotion_emb = self.emotion_embedding(emotion_id)
        emotion_features = self.emotion_encoder(emotion_emb)
        
        # Apply emotion-based modifications
        pitch_scale = torch.sigmoid(self.pitch_mod(emotion_features))
        energy_scale = torch.sigmoid(self.energy_mod(emotion_features))
        duration_scale = torch.sigmoid(self.duration_mod(emotion_features))
        
        # Scale the predictions
        pitch = pitch * pitch_scale.unsqueeze(1)
        energy = energy * energy_scale.unsqueeze(1)
        duration = duration * duration_scale.unsqueeze(1)
        
        return pitch, energy, duration

class ProsodyPredictor(nn.Module):
    """Complete prosody prediction system"""
    def __init__(self, 
                 input_size: int = 768, 
                 hidden_size: int = 256,
                 num_emotions: int = 8):
        super().__init__()
        
        # Core predictors
        self.duration_predictor = DurationPredictor(input_size, hidden_size)
        self.pitch_predictor = PitchPredictor(input_size, hidden_size)
        self.energy_predictor = EnergyPredictor(input_size, hidden_size)
        
        # Emotion adaptation
        self.emotion_adapter = EmotionalProsodyAdapter(num_emotions, hidden_size)
        
        # Sentence-level prosody
        self.sentence_encoder = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.sentence_proj = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self,
                x: torch.Tensor,
                emotion_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Predict all prosody features
        
        Args:
            x: Input features (batch, time, channels)
            emotion_ids: Optional emotion IDs for prosody modification
            
        Returns:
            Dictionary containing all prosody predictions
        """
        # Get base predictions
        duration = self.duration_predictor(x)
        pitch = self.pitch_predictor(x)
        energy = self.energy_predictor(x)
        
        # Get sentence-level features
        sentence_features, _ = self.sentence_encoder(x)
        sentence_features = self.sentence_proj(sentence_features)
        
        # Apply emotion adaptation if provided
        if emotion_ids is not None:
            pitch, energy, duration = self.emotion_adapter(
                emotion_ids, pitch, energy, duration
            )
            
        return {
            'duration': duration,
            'pitch': pitch,
            'energy': energy,
            'sentence_features': sentence_features
        }
        
    def get_emotion_embedding(self, emotion_id: torch.Tensor) -> torch.Tensor:
        """Get emotion embedding for a given emotion ID"""
        return self.emotion_adapter.emotion_embedding(emotion_id)