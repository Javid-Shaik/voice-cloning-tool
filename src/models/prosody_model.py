"""
Emotion-Aware Prosody Model
- Hierarchical prosody modeling with emotion conditioning
- Supports multiple prosody levels (phoneme, word, phrase)
- Handles emotion-specific prosody variations
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

class EmotionProsodyEncoder(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 emotion_dim: int = 3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.emotion_dim = emotion_dim
        
        # Emotion conditioning network
        self.emotion_net = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Prosody encoder
        self.encoder = nn.GRU(
            input_size=input_dim + hidden_dim,  # Features + emotion
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Output projection
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self,
                features: torch.Tensor,
                emotion_cond: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode prosody with emotion conditioning
        
        Args:
            features: Input features [batch, time, dim]
            emotion_cond: Emotion conditioning [batch, emotion_dim]
            lengths: Sequence lengths
            
        Returns:
            Encoded prosody features
        """
        # Process emotion condition
        emotion_encoding = self.emotion_net(emotion_cond)
        emotion_encoding = emotion_encoding.unsqueeze(1)  # [batch, 1, hidden]
        
        # Expand emotion to match sequence length
        emotion_encoding = emotion_encoding.expand(-1, features.size(1), -1)
        
        # Concatenate features with emotion
        inputs = torch.cat([features, emotion_encoding], dim=-1)
        
        # Pack if lengths provided
        if lengths is not None:
            inputs = nn.utils.rnn.pack_padded_sequence(
                inputs, lengths, batch_first=True, enforce_sorted=False
            )
        
        # Encode
        outputs, _ = self.encoder(inputs)
        
        # Unpack if needed
        if lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True
            )
            
        # Project
        outputs = self.proj(outputs)
        return outputs

class HierarchicalProsodyModel(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 emotion_dim: int = 3):
        super().__init__()
        
        # Feature dimension for each level
        self.phoneme_dim = input_dim
        self.word_dim = hidden_dim
        self.phrase_dim = hidden_dim * 2
        
        # Prosody encoders for each level
        self.phoneme_encoder = EmotionProsodyEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            emotion_dim=emotion_dim
        )
        
        self.word_encoder = EmotionProsodyEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            emotion_dim=emotion_dim
        )
        
        self.phrase_encoder = EmotionProsodyEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            emotion_dim=emotion_dim
        )
        
        # Level combination
        self.combiner = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
    def forward(self,
                phoneme_features: torch.Tensor,
                word_boundaries: torch.Tensor,
                phrase_boundaries: torch.Tensor,
                emotion_cond: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process hierarchical prosody with emotion
        
        Args:
            phoneme_features: Phoneme-level features
            word_boundaries: Word boundary indicators
            phrase_boundaries: Phrase boundary indicators
            emotion_cond: Emotion conditioning
            lengths: Sequence lengths
            
        Returns:
            Dictionary of prosody features at each level
        """
        # Encode each level
        phoneme_enc = self.phoneme_encoder(
            phoneme_features, emotion_cond, lengths
        )
        
        word_enc = self.word_encoder(
            phoneme_enc, emotion_cond, lengths
        )
        
        phrase_enc = self.phrase_encoder(
            word_enc, emotion_cond, lengths
        )
        
        # Combine levels
        combined = torch.cat([
            phoneme_enc,
            word_enc,
            phrase_enc
        ], dim=-1)
        
        combined = self.combiner(combined)
        
        return {
            'phoneme': phoneme_enc,
            'word': word_enc,
            'phrase': phrase_enc,
            'combined': combined
        }
    
    def get_emotion_prosody_style(self,
                                emotion_cond: torch.Tensor,
                                num_frames: int) -> torch.Tensor:
        """Generate emotion-specific prosody style"""
        # Process emotion through each level
        style_phoneme = self.phoneme_encoder.emotion_net(emotion_cond)
        style_word = self.word_encoder.emotion_net(emotion_cond)
        style_phrase = self.phrase_encoder.emotion_net(emotion_cond)
        
        # Combine styles
        style = torch.cat([
            style_phoneme,
            style_word,
            style_phrase
        ], dim=-1)
        
        style = self.combiner(style)
        
        # Expand to requested length
        style = style.unsqueeze(1).expand(-1, num_frames, -1)
        
        return style