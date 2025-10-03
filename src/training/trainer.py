"""
Training infrastructure for emotion-aware voice synthesis
- Handles data loading and preprocessing
- Implements training loops and validation
- Manages model checkpoints and logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
from pathlib import Path
import json
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 0.0001
    num_epochs: int = 100
    checkpoint_interval: int = 10
    validation_interval: int = 5
    grad_clip: float = 1.0
    emotion_weight: float = 0.5
    prosody_weight: float = 0.3
    reconstruction_weight: float = 1.0

class EmotionAwareDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 feature_extractors: Dict,
                 max_len: int = 1000):
        self.data_dir = Path(data_dir)
        self.feature_extractors = feature_extractors
        self.max_len = max_len
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Setup feature cache
        self.feature_cache = {}
        
    def _load_metadata(self) -> List[Dict]:
        """Load dataset metadata"""
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
            
        with open(metadata_path) as f:
            return json.load(f)
            
    def _extract_features(self, idx: int) -> Dict[str, torch.Tensor]:
        """Extract or load features for an item"""
        if idx in self.feature_cache:
            return self.feature_cache[idx]
            
        item = self.metadata[idx]
        audio_path = self.data_dir / item['audio_file']
        text = item['text']
        emotion = item.get('emotion', 'neutral')
        
        # Extract features
        linguistic = self.feature_extractors['linguistic'].extract_features(text)
        acoustic = self.feature_extractors['acoustic'].extract_features(audio_path)
        emotion_feats = self.feature_extractors['emotion'].extract_features(
            text, acoustic
        )
        
        features = {
            'linguistic': linguistic,
            'acoustic': acoustic,
            'emotion': emotion_feats,
            'text': text,
            'emotion_label': emotion
        }
        
        # Cache features
        self.feature_cache[idx] = features
        return features
        
    def __len__(self) -> int:
        return len(self.metadata)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._extract_features(idx)

class EmotionAwareTrainer:
    def __init__(self,
                 model: nn.Module,
                 config: TrainingConfig,
                 device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None) -> None:
        """Main training loop"""
        for epoch in range(self.config.num_epochs):
            # Training
            train_losses = self._train_epoch(train_loader)
            
            # Logging
            self.logger.info(
                f"Epoch {epoch}: train_loss={train_losses['total']:.4f}"
            )
            
            # Validation
            if val_loader and epoch % self.config.validation_interval == 0:
                val_losses = self._validate(val_loader)
                self.logger.info(
                    f"Validation loss: {val_losses['total']:.4f}"
                )
                
            # Checkpointing
            if epoch % self.config.checkpoint_interval == 0:
                self._save_checkpoint(epoch)
                
    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_losses = {
            'total': 0,
            'reconstruction': 0,
            'emotion': 0,
            'prosody': 0
        }
        
        for batch in train_loader:
            # Move to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(batch)
            
            # Calculate losses
            losses = self._compute_losses(batch, outputs)
            loss = self._combine_losses(losses)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )
            
            self.optimizer.step()
            
            # Update totals
            for k, v in losses.items():
                total_losses[k] += v.item()
                
        # Average losses
        for k in total_losses:
            total_losses[k] /= len(train_loader)
            
        return total_losses
    
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation pass"""
        self.model.eval()
        total_losses = {
            'total': 0,
            'reconstruction': 0,
            'emotion': 0,
            'prosody': 0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch)
                
                # Calculate losses
                losses = self._compute_losses(batch, outputs)
                
                # Update totals
                for k, v in losses.items():
                    total_losses[k] += v.item()
                    
        # Average losses
        for k in total_losses:
            total_losses[k] /= len(val_loader)
            
        return total_losses
    
    def _compute_losses(self,
                       batch: Dict[str, torch.Tensor],
                       outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute all loss components"""
        losses = {}
        
        # Reconstruction loss
        losses['reconstruction'] = nn.MSELoss()(
            outputs['reconstructed'],
            batch['acoustic']['mel']
        )
        
        # Emotion loss
        losses['emotion'] = nn.CrossEntropyLoss()(
            outputs['emotion_logits'],
            batch['emotion_label']
        )
        
        # Prosody loss
        losses['prosody'] = nn.MSELoss()(
            outputs['prosody'],
            batch['acoustic']['f0']
        )
        
        return losses
    
    def _combine_losses(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine loss components with weights"""
        total_loss = (
            self.config.reconstruction_weight * losses['reconstruction'] +
            self.config.emotion_weight * losses['emotion'] +
            self.config.prosody_weight * losses['prosody']
        )
        return total_loss
    
    def _save_checkpoint(self, epoch: int) -> None:
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        path = f"checkpoints/model_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")