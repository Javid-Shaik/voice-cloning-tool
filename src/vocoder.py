"""
High-Fidelity Neural Vocoder Module
- Implements HiFi-GAN vocoder for high-quality audio synthesis
- Includes multi-band processing and phase reconstruction
"""

import torch
import torch.nn as nn
import torchaudio
from typing import Optional, Tuple

class HiFiGANVocoder(nn.Module):
    def __init__(self, 
                 model_name: str = "universal_large",
                 sample_rate: int = 48000):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Load pretrained HiFi-GAN model
        self.model = torch.hub.load("bshall/hifigan:main", model_name)
        self.model.eval()
        
        # Multi-band processing components
        self.num_bands = 4
        self.band_encoders = nn.ModuleList([
            nn.Conv1d(1, 64, kernel_size=7, padding=3)
            for _ in range(self.num_bands)
        ])
        
        # Phase reconstruction network
        self.phase_net = nn.Sequential(
            nn.Conv1d(64 * self.num_bands, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 2, 3, padding=1)  # Real and imaginary components
        )
        
    def split_bands(self, mel_spec: torch.Tensor) -> Tuple[torch.Tensor]:
        """Split mel spectrogram into frequency bands"""
        band_size = mel_spec.size(1) // self.num_bands
        bands = []
        for i in range(self.num_bands):
            start_idx = i * band_size
            end_idx = start_idx + band_size
            band = mel_spec[:, start_idx:end_idx, :]
            bands.append(band)
        return tuple(bands)
    
    def process_bands(self, bands: Tuple[torch.Tensor]) -> torch.Tensor:
        """Process each frequency band separately"""
        band_features = []
        for band, encoder in zip(bands, self.band_encoders):
            # Reshape for 1D convolution
            B, F, T = band.size()
            band_flat = band.reshape(-1, 1, T)  # (B*F, 1, T)
            
            # Process through band encoder
            features = encoder(band_flat)  # (B*F, 64, T)
            features = features.reshape(B, F, 64, T)
            band_features.append(features)
            
        # Concatenate band features
        return torch.cat(band_features, dim=2)  # (B, F, 64*num_bands, T)
    
    def reconstruct_phase(self, 
                         band_features: torch.Tensor, 
                         mel_spec: torch.Tensor) -> torch.Tensor:
        """Reconstruct phase information from band features"""
        B, F, C, T = band_features.size()
        features_flat = band_features.reshape(B, F*C, T)
        
        # Generate phase components
        phase_components = self.phase_net(features_flat)
        phase = torch.atan2(phase_components[:, 1:2], phase_components[:, 0:1])
        
        return phase
    
    @torch.no_grad()
    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Convert mel spectrogram to audio waveform
        
        Args:
            mel_spec: Tensor of shape (batch_size, n_mels, time)
            
        Returns:
            Tensor of shape (batch_size, time_samples) containing audio waveform
        """
        # Split into frequency bands
        bands = self.split_bands(mel_spec)
        
        # Process each band
        band_features = self.process_bands(bands)
        
        # Reconstruct phase information
        phase = self.reconstruct_phase(band_features, mel_spec)
        
        # Combine magnitude and phase
        complex_spec = torch.polar(mel_spec, phase)
        
        # Generate audio with HiFi-GAN
        audio = self.model(complex_spec)
        
        return audio
    
    def save_audio(self, 
                  audio: torch.Tensor,
                  filename: str,
                  normalize: bool = True):
        """Save audio tensor to file"""
        if normalize:
            audio = audio / torch.abs(audio).max()
        torchaudio.save(filename, audio.cpu(), self.sample_rate)