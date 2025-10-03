"""
Emotional Expression Integration Module
- Handles emotional style transfer
- Manages emotion-specific voice characteristics
- Implements continuous emotion interpolation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import numpy as np

class EmotionalStyleEncoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 style_dim: int,
                 emotion_dim: int = 3):
        super().__init__()
        
        self.input_dim = input_dim
        self.style_dim = style_dim
        self.emotion_dim = emotion_dim
        
        # Style encoder
        self.style_encoder = nn.Sequential(
            nn.Linear(input_dim, style_dim * 2),
            nn.ReLU(),
            nn.Linear(style_dim * 2, style_dim)
        )
        
        # Emotion encoder
        self.emotion_encoder = nn.Sequential(
            nn.Linear(emotion_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim)
        )
        
        # Style-emotion fusion
        self.fusion = nn.Sequential(
            nn.Linear(style_dim * 2, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim)
        )
        
    def forward(self,
                features: torch.Tensor,
                emotion_cond: torch.Tensor) -> torch.Tensor:
        """
        Encode emotional style
        
        Args:
            features: Input features [batch, time, dim]
            emotion_cond: Emotion conditioning [batch, emotion_dim]
            
        Returns:
            Emotional style encoding
        """
        # Encode style
        style = self.style_encoder(features)
        
        # Encode emotion
        emotion = self.emotion_encoder(emotion_cond)
        
        # Fuse style and emotion
        combined = torch.cat([style, emotion], dim=-1)
        emotional_style = self.fusion(combined)
        
        return emotional_style

class EmotionalStyleTransfer(nn.Module):
    def __init__(self,
                 input_dim: int,
                 style_dim: int,
                 emotion_dim: int = 3):
        super().__init__()
        
        self.style_encoder = EmotionalStyleEncoder(
            input_dim=input_dim,
            style_dim=style_dim,
            emotion_dim=emotion_dim
        )
        
        # Style decoder
        self.decoder = nn.Sequential(
            nn.Linear(style_dim + emotion_dim, input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim)
        )
        
        # Adaptive Instance Normalization
        self.adain = AdaptiveInstanceNorm()
        
    def forward(self,
                content: torch.Tensor,
                style: torch.Tensor,
                emotion_cond: torch.Tensor) -> torch.Tensor:
        """
        Apply emotional style transfer
        
        Args:
            content: Content features
            style: Style features
            emotion_cond: Emotion conditioning
            
        Returns:
            Style transferred features
        """
        # Encode emotional style
        style_code = self.style_encoder(style, emotion_cond)
        
        # Apply style transfer
        transferred = self.adain(content, style_code)
        
        # Decode with emotion
        output = self.decoder(
            torch.cat([transferred, emotion_cond], dim=-1)
        )
        
        return output

class AdaptiveInstanceNorm(nn.Module):
    def forward(self,
                content: torch.Tensor,
                style_code: torch.Tensor) -> torch.Tensor:
        """Apply Adaptive Instance Normalization"""
        # Content statistics
        c_mean = torch.mean(content, dim=(2, 3), keepdim=True)
        c_std = torch.std(content, dim=(2, 3), keepdim=True)
        
        # Normalize content
        normalized = (content - c_mean) / (c_std + 1e-8)
        
        # Apply style
        return normalized * style_code

class EmotionalExpressionModule:
    def __init__(self,
                 input_dim: int,
                 style_dim: int,
                 emotion_dim: int = 3):
        self.style_transfer = EmotionalStyleTransfer(
            input_dim=input_dim,
            style_dim=style_dim,
            emotion_dim=emotion_dim
        )
        
        # Emotion transition handler
        self.transition = EmotionTransitionHandler(emotion_dim)
        
    def apply_emotion(self,
                     features: Dict[str, torch.Tensor],
                     target_emotion: torch.Tensor,
                     style_ref: Optional[Dict[str, torch.Tensor]] = None
                     ) -> Dict[str, torch.Tensor]:
        """
        Apply emotional expression to features
        
        Args:
            features: Input features
            target_emotion: Target emotion vector
            style_ref: Optional reference style features
            
        Returns:
            Emotion-modified features
        """
        modified = {}
        
        # Apply style transfer if reference provided
        if style_ref is not None:
            for feat_type, feat in features.items():
                if feat_type in style_ref:
                    modified[feat_type] = self.style_transfer(
                        feat,
                        style_ref[feat_type],
                        target_emotion
                    )
                else:
                    modified[feat_type] = feat
        else:
            modified = features
            
        # Apply direct emotion modifications
        modified = self._modify_features(modified, target_emotion)
        
        return modified
    
    def _modify_features(self,
                        features: Dict[str, torch.Tensor],
                        emotion: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply emotion-specific modifications"""
        modified = {}
        
        # Extract emotion components
        valence, arousal, dominance = emotion.split(1, dim=-1)
        
        for feat_type, feat in features.items():
            if feat_type == 'f0':
                # Modify pitch characteristics
                mod = self._modify_pitch(feat, valence, arousal, dominance)
            elif feat_type == 'energy':
                # Modify energy characteristics
                mod = self._modify_energy(feat, valence, arousal, dominance)
            elif feat_type == 'duration':
                # Modify duration characteristics
                mod = self._modify_duration(feat, valence, arousal)
            else:
                mod = feat
                
            modified[feat_type] = mod
            
        return modified
    
    def _modify_pitch(self,
                     f0: torch.Tensor,
                     valence: torch.Tensor,
                     arousal: torch.Tensor,
                     dominance: torch.Tensor) -> torch.Tensor:
        """Modify pitch based on emotion"""
        # Base scaling
        scale = 1.0 + (0.2 * arousal) + (0.1 * dominance)
        f0 = f0 * scale
        
        # Add variations based on valence
        if valence > 0:
            variations = torch.sin(
                torch.linspace(0, 10*np.pi, f0.size(-1))
            ).to(f0.device)
            f0 = f0 + (variations * 0.1 * valence)
            
        return f0
    
    def _modify_energy(self,
                      energy: torch.Tensor,
                      valence: torch.Tensor,
                      arousal: torch.Tensor,
                      dominance: torch.Tensor) -> torch.Tensor:
        """Modify energy based on emotion"""
        # Base scaling
        scale = 1.0 + (0.3 * arousal) + (0.2 * dominance)
        energy = energy * scale
        
        # Dynamic range modification
        if valence < 0:
            # Compress dynamic range for negative valence
            energy = torch.tanh(energy)
        else:
            # Expand dynamic range for positive valence
            energy = energy * (1.0 + 0.2 * valence)
            
        return energy
    
    def _modify_duration(self,
                        duration: torch.Tensor,
                        valence: torch.Tensor,
                        arousal: torch.Tensor) -> torch.Tensor:
        """Modify duration based on emotion"""
        # Speed up for high arousal, slow down for low arousal
        scale = 1.0 / (1.0 + 0.3 * arousal)
        duration = duration * scale
        
        # Add variations based on valence
        if valence != 0:
            variations = 1.0 + 0.1 * torch.randn_like(duration) * valence
            duration = duration * variations
            
        return duration

class EmotionTransitionHandler:
    def __init__(self, emotion_dim: int):
        self.emotion_dim = emotion_dim
        
    def interpolate_emotions(self,
                           start_emotion: torch.Tensor,
                           end_emotion: torch.Tensor,
                           num_steps: int) -> torch.Tensor:
        """Generate smooth emotion transition"""
        weights = torch.linspace(0, 1, num_steps)
        
        transitions = []
        for w in weights:
            # Linear interpolation
            emotion = start_emotion * (1 - w) + end_emotion * w
            transitions.append(emotion)
            
        return torch.stack(transitions)
    
    def generate_emotion_sequence(self,
                                emotions: List[torch.Tensor],
                                durations: List[int],
                                transition_lengths: List[int]
                                ) -> torch.Tensor:
        """Generate emotion sequence with smooth transitions"""
        sequence = []
        
        for i in range(len(emotions) - 1):
            # Current emotion segment
            sequence.extend([emotions[i]] * (durations[i] - transition_lengths[i]))
            
            # Transition to next emotion
            transition = self.interpolate_emotions(
                emotions[i],
                emotions[i + 1],
                transition_lengths[i]
            )
            sequence.extend(transition)
            
        # Add final emotion segment
        sequence.extend([emotions[-1]] * durations[-1])
        
        return torch.stack(sequence)