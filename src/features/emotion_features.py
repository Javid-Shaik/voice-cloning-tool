"""
Emotion Feature Extraction Module
- Extracts emotion-specific features from text and audio
- Combines linguistic and acoustic features for emotion
- Supports continuous emotion representation
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import pipeline

class EmotionFeatureExtractor:
    def __init__(self, emotion_model: str = "j-hartmann/emotion-english-distilroberta-base"):
        # Initialize emotion classification model
        self.emotion_classifier = pipeline("text-classification", 
                                        model=emotion_model, 
                                        return_all_scores=True)
        
        # Basic emotion dimensions (valence, arousal, dominance)
        self.emotion_dims = ['valence', 'arousal', 'dominance']
        
        # Emotion to VAD mapping (pre-defined based on research)
        self.emotion_vad = {
            'joy': [0.8, 0.7, 0.6],      # High valence, high arousal
            'anger': [-0.8, 0.7, 0.7],    # Low valence, high arousal, high dominance
            'sadness': [-0.7, -0.6, -0.5],# Low valence, low arousal, low dominance
            'fear': [-0.7, 0.7, -0.6],    # Low valence, high arousal, low dominance
            'surprise': [0.4, 0.8, 0.0],  # Mid valence, high arousal
            'neutral': [0.0, 0.0, 0.0]    # Neutral across dimensions
        }
        
    def extract_features(self, 
                        text: str,
                        acoustic_features: Optional[Dict[str, torch.Tensor]] = None
                        ) -> Dict[str, torch.Tensor]:
        """
        Extract comprehensive emotion features from text and optional acoustic features
        
        Args:
            text: Input text
            acoustic_features: Optional acoustic features for multimodal analysis
            
        Returns:
            Dictionary containing:
            - emotion_probs: Emotion class probabilities
            - emotion_vad: Valence-Arousal-Dominance values
            - emotion_acoustic: Emotion-relevant acoustic features (if provided)
        """
        # Get emotion classifications
        emotions = self.emotion_classifier(text)[0]
        emotion_probs = torch.tensor([e['score'] for e in emotions])
        
        # Convert to VAD space
        vad_values = self._emotions_to_vad(emotions)
        
        # Extract acoustic emotion features if provided
        acoustic_emotion = None
        if acoustic_features is not None:
            acoustic_emotion = self._extract_acoustic_emotion(acoustic_features)
            
        return {
            'emotion_probs': emotion_probs,
            'emotion_vad': vad_values,
            'emotion_acoustic': acoustic_emotion
        }
    
    def _emotions_to_vad(self, emotions) -> torch.Tensor:
        """Convert emotion probabilities to VAD space"""
        vad = np.zeros(3)  # [valence, arousal, dominance]
        
        # Weighted sum of VAD values based on emotion probabilities
        for emotion in emotions:
            label = emotion['label'].lower()
            if label in self.emotion_vad:
                vad += np.array(self.emotion_vad[label]) * emotion['score']
                
        # Normalize
        vad = np.clip(vad, -1.0, 1.0)
        return torch.tensor(vad)
    
    def _extract_acoustic_emotion(self, acoustic_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract emotion-relevant features from acoustic features"""
        emotion_features = []
        
        if 'f0' in acoustic_features:
            # F0 statistics for emotion
            f0 = acoustic_features['f0']
            f0_stats = torch.tensor([
                torch.mean(f0),
                torch.std(f0),
                torch.max(f0) - torch.min(f0),  # Range
                torch.median(f0)
            ])
            emotion_features.append(f0_stats)
            
        if 'energy' in acoustic_features:
            # Energy contour features
            energy = acoustic_features['energy']
            energy_stats = torch.tensor([
                torch.mean(energy),
                torch.std(energy),
                torch.max(energy) - torch.min(energy)
            ])
            emotion_features.append(energy_stats)
            
        if 'spectral' in acoustic_features:
            # Spectral features for emotion
            spectral = acoustic_features['spectral']
            spectral_stats = torch.tensor([
                torch.mean(spectral, dim=0),
                torch.std(spectral, dim=0)
            ])
            emotion_features.append(spectral_stats.flatten())
            
        # Combine all features
        if emotion_features:
            return torch.cat(emotion_features)
        return None
    
    def get_emotion_control_vector(self, target_emotion: str, intensity: float = 1.0) -> torch.Tensor:
        """
        Generate control vector for target emotion with specified intensity
        
        Args:
            target_emotion: Target emotion label
            intensity: Emotion intensity (0.0 to 1.0)
            
        Returns:
            Control vector for emotion synthesis
        """
        # Get base VAD values for emotion
        if target_emotion.lower() in self.emotion_vad:
            vad = np.array(self.emotion_vad[target_emotion.lower()])
        else:
            vad = np.zeros(3)  # Neutral if emotion unknown
            
        # Apply intensity
        vad = vad * intensity
        
        # Convert to tensor
        return torch.tensor(vad)
    
    def interpolate_emotions(self, 
                           emotion1: str,
                           emotion2: str,
                           weight: float) -> torch.Tensor:
        """
        Interpolate between two emotions
        
        Args:
            emotion1: First emotion
            emotion2: Second emotion
            weight: Interpolation weight (0.0 = emotion1, 1.0 = emotion2)
            
        Returns:
            Interpolated VAD values
        """
        # Get VAD values
        vad1 = np.array(self.emotion_vad.get(emotion1.lower(), [0, 0, 0]))
        vad2 = np.array(self.emotion_vad.get(emotion2.lower(), [0, 0, 0]))
        
        # Linear interpolation
        vad = vad1 * (1 - weight) + vad2 * weight
        return torch.tensor(vad)