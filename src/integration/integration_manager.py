"""
Integration Manager for Emotion-Aware Voice Cloning
- Orchestrates all components
- Manages feature pipeline
- Handles cross-component communication
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from features.linguistic_features import LinguisticFeatureExtractor
from features.emotion_features import EmotionFeatureExtractor
from features.acoustic_normalizer import AcousticNormalizer
from audio.audio_processor import AudioProcessor
from emotion.emotion_expression import EmotionalExpressionModule
from context.context_awareness import ContextManager
from models.prosody_model import HierarchicalProsodyModel
from evaluation.quality_assurance import QualityAssurance

class IntegrationManager:
    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        
        # Initialize all components
        self.linguistic_extractor = LinguisticFeatureExtractor()
        self.emotion_extractor = EmotionFeatureExtractor()
        self.acoustic_normalizer = AcousticNormalizer()
        self.audio_processor = AudioProcessor()
        self.emotion_expression = EmotionalExpressionModule(
            input_dim=256,
            style_dim=128,
            emotion_dim=3
        )
        self.context_manager = ContextManager()
        self.prosody_model = HierarchicalProsodyModel(
            input_dim=256,
            hidden_dim=128,
            emotion_dim=3
        ).to(device)
        
        self.quality_assurance = QualityAssurance()
        
    def process_text(self,
                    text: str,
                    emotion: Optional[str] = None,
                    context: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """Process input text through feature extraction pipeline"""
        # Extract linguistic features
        linguistic_features = self.linguistic_extractor.extract_features(text)
        
        # Extract emotion features
        emotion_features = self.emotion_extractor.extract_features(text)
        
        # Update context if provided
        if context:
            self.context_manager.update_context(
                speaker_emotion=emotion_features['emotion_vad'],
                text=text,
                situation=context.get('situation'),
                social=context.get('social')
            )
            
        # Get context-aware features
        context_embedding = self.context_manager.get_context_embedding()
        
        # Generate prosody with emotion and context
        prosody_features = self.prosody_model(
            linguistic_features['bert_embeddings'],
            linguistic_features['pos_features'],
            linguistic_features['dep_features'],
            emotion_features['emotion_vad'],
            context_embedding
        )
        
        return {
            'linguistic': linguistic_features,
            'emotion': emotion_features,
            'prosody': prosody_features,
            'context': context_embedding
        }
        
    def process_audio(self,
                     audio: torch.Tensor,
                     features: Dict[str, torch.Tensor],
                     emotion: Optional[str] = None) -> torch.Tensor:
        """Process audio through enhancement pipeline"""
        # Normalize audio features
        audio_features = self.audio_processor.process_audio(audio)
        normalized = self.acoustic_normalizer.normalize(audio_features)
        
        # Apply emotion expression
        if emotion:
            emotion_vector = self.emotion_extractor.get_emotion_control_vector(emotion)
            modified = self.emotion_expression.apply_emotion(
                normalized,
                emotion_vector
            )
        else:
            modified = normalized
            
        # Apply context-aware modifications
        modified = self.context_manager.adapt_features(modified)
        
        # Convert back to waveform
        output = self.audio_processor.inverse_transform(
            modified['mel_spectrogram'],
            modified['f0']
        )
        
        return output
        
    def evaluate_quality(self,
                        generated: torch.Tensor,
                        reference: torch.Tensor) -> Dict[str, float]:
        """Evaluate synthesis quality"""
        return self.quality_assurance.evaluate_synthesis(
            generated,
            reference
        )
        
    @torch.no_grad()
    def generate_speech(self,
                       text: str,
                       emotion: Optional[str] = None,
                       context: Optional[Dict] = None,
                       reference_audio: Optional[torch.Tensor] = None
                       ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Generate emotion-aware speech with quality metrics"""
        # Process text
        features = self.process_text(text, emotion, context)
        
        # Generate base audio
        audio = self.prosody_model.generate(features)
        
        # Process audio
        output = self.process_audio(audio, features, emotion)
        
        # Evaluate quality if reference provided
        metrics = {}
        if reference_audio is not None:
            metrics = self.evaluate_quality(output, reference_audio)
            
        return output, metrics