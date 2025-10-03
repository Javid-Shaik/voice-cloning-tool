"""
Acoustic Feature Normalization Module
- Handles normalization of acoustic features
- Supports online and offline normalization
- Provides emotion-aware feature scaling
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class NormalizationStats:
    mean: torch.Tensor
    std: torch.Tensor
    min_val: torch.Tensor
    max_val: torch.Tensor

class AcousticNormalizer:
    def __init__(self):
        self.stats = {}
        self.online_stats = {}
        
    def fit(self, features: Dict[str, torch.Tensor]) -> None:
        """
        Calculate normalization statistics from features
        
        Args:
            features: Dictionary of feature tensors
        """
        for name, feat in features.items():
            self.stats[name] = NormalizationStats(
                mean=torch.mean(feat, dim=0),
                std=torch.std(feat, dim=0),
                min_val=torch.min(feat, dim=0)[0],
                max_val=torch.max(feat, dim=0)[0]
            )
            
    def normalize(self, 
                 features: Dict[str, torch.Tensor],
                 method: str = 'z_score',
                 emotion_aware: bool = True) -> Dict[str, torch.Tensor]:
        """
        Normalize acoustic features
        
        Args:
            features: Dictionary of feature tensors
            method: Normalization method ('z_score', 'min_max', 'robust')
            emotion_aware: Whether to apply emotion-specific normalization
            
        Returns:
            Dictionary of normalized features
        """
        normalized = {}
        
        for name, feat in features.items():
            if name not in self.stats:
                continue
                
            stats = self.stats[name]
            
            if method == 'z_score':
                normalized[name] = (feat - stats.mean) / (stats.std + 1e-8)
            elif method == 'min_max':
                normalized[name] = (feat - stats.min_val) / (stats.max_val - stats.min_val + 1e-8)
            elif method == 'robust':
                q1 = torch.quantile(feat, 0.25, dim=0)
                q3 = torch.quantile(feat, 0.75, dim=0)
                iqr = q3 - q1
                normalized[name] = (feat - q1) / (iqr + 1e-8)
                
            if emotion_aware:
                normalized[name] = self._apply_emotion_scaling(normalized[name], name)
                
        return normalized
    
    def _apply_emotion_scaling(self, 
                             features: torch.Tensor,
                             feature_type: str) -> torch.Tensor:
        """Apply emotion-specific scaling factors"""
        # Feature-specific emotion scaling
        if feature_type == 'f0':
            # Preserve emotion-specific F0 variations
            return features * 1.2  # Enhance F0 dynamics
        elif feature_type == 'energy':
            # Preserve emotion-specific energy patterns
            return features * 1.1  # Enhance energy dynamics
        elif feature_type == 'spectral':
            # Preserve spectral characteristics
            return features * 1.0  # Keep spectral features as is
        return features
    
    def update_online_stats(self, features: Dict[str, torch.Tensor]) -> None:
        """Update running statistics for online normalization"""
        for name, feat in features.items():
            if name not in self.online_stats:
                self.online_stats[name] = {
                    'count': 0,
                    'mean': 0,
                    'M2': 0,
                }
            
            stats = self.online_stats[name]
            stats['count'] += 1
            delta = feat - stats['mean']
            stats['mean'] += delta / stats['count']
            delta2 = feat - stats['mean']
            stats['M2'] += delta * delta2
            
    def get_online_stats(self, feature_name: str) -> Optional[NormalizationStats]:
        """Get current online normalization statistics"""
        if feature_name not in self.online_stats:
            return None
            
        stats = self.online_stats[feature_name]
        count = stats['count']
        
        if count < 2:
            return None
            
        return NormalizationStats(
            mean=stats['mean'],
            std=torch.sqrt(stats['M2'] / (count - 1)),
            min_val=torch.min(stats['mean']),
            max_val=torch.max(stats['mean'])
        )