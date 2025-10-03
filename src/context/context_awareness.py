"""
Context Awareness Module
- Analyzes and adapts to conversation context
- Handles contextual emotional transitions
- Manages situational awareness
- Supports context-dependent prosody
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer, AutoModel

@dataclass
class ConversationContext:
    speaker_emotion: torch.Tensor  # Current speaker emotion
    listener_emotion: Optional[torch.Tensor]  # Listener's emotion if available
    conversation_history: List[Dict]  # Previous utterances and emotions
    situational_context: Dict[str, float]  # Environmental/situational factors
    social_context: Dict[str, float]  # Social relationship factors
    
class ContextAnalyzer:
    def __init__(self,
                 context_embedding_dim: int = 256,
                 history_length: int = 10,
                 bert_model: str = "bert-base-uncased"):
        self.context_embedding_dim = context_embedding_dim
        self.history_length = history_length
        
        # Initialize BERT for context understanding
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self.bert = AutoModel.from_pretrained(bert_model)
        
        # Context embedding networks
        self.emotion_context_net = nn.Sequential(
            nn.Linear(6, 64),  # Speaker + Listener emotions
            nn.ReLU(),
            nn.Linear(64, context_embedding_dim)
        )
        
        self.situation_context_net = nn.Sequential(
            nn.Linear(32, 128),  # Situational features
            nn.ReLU(),
            nn.Linear(128, context_embedding_dim)
        )
        
        self.social_context_net = nn.Sequential(
            nn.Linear(16, 64),  # Social relationship features
            nn.ReLU(),
            nn.Linear(64, context_embedding_dim)
        )
        
        # Context fusion network
        self.context_fusion = nn.Sequential(
            nn.Linear(context_embedding_dim * 4, context_embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(context_embedding_dim * 2, context_embedding_dim)
        )
        
    def analyze_context(self, context: ConversationContext) -> torch.Tensor:
        """
        Analyze conversation context and generate context embedding
        
        Args:
            context: Conversation context information
            
        Returns:
            Context embedding tensor
        """
        # Process emotional context
        emotion_context = self._process_emotional_context(
            context.speaker_emotion,
            context.listener_emotion
        )
        
        # Process conversation history
        history_context = self._process_conversation_history(
            context.conversation_history
        )
        
        # Process situational context
        situation_context = self._process_situational_context(
            context.situational_context
        )
        
        # Process social context
        social_context = self._process_social_context(
            context.social_context
        )
        
        # Fuse all context embeddings
        context_embedding = self.context_fusion(torch.cat([
            emotion_context,
            history_context,
            situation_context,
            social_context
        ], dim=-1))
        
        return context_embedding
    
    def _process_emotional_context(self,
                                 speaker_emotion: torch.Tensor,
                                 listener_emotion: Optional[torch.Tensor]
                                 ) -> torch.Tensor:
        """Process emotional context"""
        if listener_emotion is None:
            # If no listener emotion, duplicate speaker emotion
            listener_emotion = speaker_emotion
            
        # Combine speaker and listener emotions
        emotion_context = torch.cat([speaker_emotion, listener_emotion])
        return self.emotion_context_net(emotion_context)
    
    def _process_conversation_history(self,
                                    history: List[Dict]) -> torch.Tensor:
        """Process conversation history using BERT"""
        # Prepare history text
        history_text = " [SEP] ".join(
            [f"{h['speaker']}: {h['text']}" for h in history[-self.history_length:]]
        )
        
        # Get BERT embeddings
        inputs = self.tokenizer(
            history_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.bert(**inputs)
            
        # Use [CLS] token embedding as history context
        return outputs.last_hidden_state[0, 0]
    
    def _process_situational_context(self,
                                   situation: Dict[str, float]) -> torch.Tensor:
        """Process situational context"""
        # Convert situation dict to tensor
        situation_features = torch.tensor([
            situation.get(factor, 0.0) for factor in self.get_situation_factors()
        ])
        
        return self.situation_context_net(situation_features)
    
    def _process_social_context(self,
                              social: Dict[str, float]) -> torch.Tensor:
        """Process social relationship context"""
        # Convert social context dict to tensor
        social_features = torch.tensor([
            social.get(factor, 0.0) for factor in self.get_social_factors()
        ])
        
        return self.social_context_net(social_features)
    
    @staticmethod
    def get_situation_factors() -> Set[str]:
        """Get list of supported situational factors"""
        return {
            'formality',
            'urgency',
            'privacy',
            'noise_level',
            'time_of_day',
            'location_type',
            'crowd_density',
            'environmental_stress'
        }
    
    @staticmethod
    def get_social_factors() -> Set[str]:
        """Get list of supported social relationship factors"""
        return {
            'familiarity',
            'power_dynamic',
            'emotional_bond',
            'trust_level'
        }

class ContextualProsodyAdapter:
    def __init__(self, context_dim: int):
        self.context_dim = context_dim
        
        # Context-to-prosody mapping
        self.prosody_mapper = nn.Sequential(
            nn.Linear(context_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
    def adapt_prosody(self,
                     prosody_features: Dict[str, torch.Tensor],
                     context_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Adapt prosody features based on context
        
        Args:
            prosody_features: Original prosody features
            context_embedding: Context embedding tensor
            
        Returns:
            Context-adapted prosody features
        """
        # Generate prosody modifications
        modifications = self.prosody_mapper(context_embedding)
        
        # Split modifications for different features
        f0_mod, energy_mod, duration_mod, _ = torch.split(modifications, [8, 8, 8, 8])
        
        # Apply modifications
        adapted = {}
        for name, feature in prosody_features.items():
            if name == 'f0':
                adapted[name] = self._modify_f0(feature, f0_mod)
            elif name == 'energy':
                adapted[name] = self._modify_energy(feature, energy_mod)
            elif name == 'duration':
                adapted[name] = self._modify_duration(feature, duration_mod)
            else:
                adapted[name] = feature
                
        return adapted
    
    def _modify_f0(self,
                   f0: torch.Tensor,
                   modification: torch.Tensor) -> torch.Tensor:
        """Modify F0 based on context"""
        # Extract modification parameters
        scale, shift, variance, range_mod = torch.split(modification[:4], 1)
        
        # Apply modifications
        modified = f0 * (1.0 + scale)  # Scale
        modified = modified + shift  # Shift
        
        # Modify variance
        if variance != 0:
            std = torch.std(modified)
            modified = modified * (1.0 + variance * std)
            
        # Modify range
        if range_mod != 0:
            min_f0, max_f0 = torch.min(modified), torch.max(modified)
            modified = (modified - min_f0) * (1.0 + range_mod) + min_f0
            
        return modified
    
    def _modify_energy(self,
                      energy: torch.Tensor,
                      modification: torch.Tensor) -> torch.Tensor:
        """Modify energy based on context"""
        # Extract modification parameters
        scale, shift, dynamics, range_mod = torch.split(modification[:4], 1)
        
        # Apply modifications
        modified = energy * (1.0 + scale)  # Scale
        modified = modified + shift  # Shift
        
        # Modify dynamics
        if dynamics != 0:
            std = torch.std(modified)
            modified = modified * (1.0 + dynamics * std)
            
        # Modify range
        if range_mod != 0:
            min_e, max_e = torch.min(modified), torch.max(modified)
            modified = (modified - min_e) * (1.0 + range_mod) + min_e
            
        return modified
    
    def _modify_duration(self,
                        duration: torch.Tensor,
                        modification: torch.Tensor) -> torch.Tensor:
        """Modify duration based on context"""
        # Extract modification parameters
        scale, shift, variance, _ = torch.split(modification[:4], 1)
        
        # Apply modifications
        modified = duration * (1.0 + scale)  # Scale
        modified = modified + shift  # Shift
        
        # Modify variance
        if variance != 0:
            std = torch.std(modified)
            modified = modified * (1.0 + variance * std)
            
        return modified

class ContextManager:
    def __init__(self,
                 context_embedding_dim: int = 256,
                 history_length: int = 10):
        self.context_analyzer = ContextAnalyzer(
            context_embedding_dim=context_embedding_dim,
            history_length=history_length
        )
        
        self.prosody_adapter = ContextualProsodyAdapter(
            context_dim=context_embedding_dim
        )
        
        self.current_context = None
        self.conversation_history = []
        
    def update_context(self,
                      speaker_emotion: torch.Tensor,
                      text: str,
                      listener_emotion: Optional[torch.Tensor] = None,
                      situation: Optional[Dict[str, float]] = None,
                      social: Optional[Dict[str, float]] = None) -> None:
        """Update conversation context"""
        # Update history
        self.conversation_history.append({
            'speaker': 'system',
            'text': text,
            'emotion': speaker_emotion
        })
        
        # Create context object
        self.current_context = ConversationContext(
            speaker_emotion=speaker_emotion,
            listener_emotion=listener_emotion,
            conversation_history=self.conversation_history,
            situational_context=situation or {},
            social_context=social or {}
        )
        
    def get_context_embedding(self) -> torch.Tensor:
        """Get current context embedding"""
        if self.current_context is None:
            raise ValueError("Context not initialized")
            
        return self.context_analyzer.analyze_context(self.current_context)
    
    def adapt_features(self,
                      features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Adapt features based on current context"""
        if self.current_context is None:
            return features
            
        context_embedding = self.get_context_embedding()
        return self.prosody_adapter.adapt_prosody(features, context_embedding)