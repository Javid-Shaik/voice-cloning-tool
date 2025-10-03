"""
Acoustic Feature Extraction Module
- Extracts F0 (pitch), energy, and other acoustic features from audio
- Provides real-time feature extraction for streaming synthesis
- Optimized for voice cloning and emotion analysis
"""

import numpy as np
import torch
import librosa
import pyworld as pw
from typing import Dict, Tuple, Optional

class AcousticFeatureExtractor:
    def __init__(self, sample_rate: int = 22050, hop_length: int = 256):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.f0_min = 50  # Minimum F0 frequency
        self.f0_max = 800  # Maximum F0 frequency
        
    def extract_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive acoustic features from audio
        
        Args:
            audio: Audio waveform (mono)
            
        Returns:
            Dictionary containing:
            - f0: Fundamental frequency contour
            - energy: Frame-level energy
            - voiced_mask: Binary voiced/unvoiced decision
            - spectral: Spectral features (MFCCs, mel spectrogram)
        """
        # Ensure float32 audio
        audio = audio.astype(np.float64)
        
        # Extract F0 using WORLD
        _f0, t = pw.dio(audio, self.sample_rate)  # Raw F0 extraction
        f0 = pw.stonemask(audio, _f0, t, self.sample_rate)  # F0 refinement
        
        # Get spectral envelope and aperiodicity
        sp = pw.cheaptrick(audio, f0, t, self.sample_rate)
        ap = pw.d4c(audio, f0, t, self.sample_rate)
        
        # Compute energy
        energy = np.sqrt(np.sum(sp * sp, axis=1))
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio.astype(np.float32), 
            sr=self.sample_rate,
            n_mels=80,
            hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio.astype(np.float32),
            sr=self.sample_rate,
            n_mfcc=13,
            hop_length=self.hop_length
        )
        
        # Create voiced mask
        voiced_mask = (f0 > self.f0_min) & (f0 < self.f0_max)
        
        return {
            'f0': f0,
            'energy': energy,
            'voiced_mask': voiced_mask,
            'mel_spectrogram': mel_spec_db,
            'mfcc': mfcc,
            'spectral_envelope': sp,
            'aperiodicity': ap
        }
    
    def extract_emotion_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract features specifically relevant for emotion detection
        
        Args:
            audio: Audio waveform
            
        Returns:
            Dictionary of emotion-relevant features
        """
        features = self.extract_features(audio)
        
        # Compute statistics over F0 contour
        voiced_f0 = features['f0'][features['voiced_mask']]
        f0_stats = {
            'f0_mean': np.mean(voiced_f0),
            'f0_std': np.std(voiced_f0),
            'f0_range': np.ptp(voiced_f0),
            'f0_slope': np.polyfit(np.arange(len(voiced_f0)), voiced_f0, 1)[0]
        }
        
        # Energy dynamics
        energy = features['energy']
        energy_stats = {
            'energy_mean': np.mean(energy),
            'energy_std': np.std(energy),
            'energy_range': np.ptp(energy),
            'energy_slope': np.polyfit(np.arange(len(energy)), energy, 1)[0]
        }
        
        # Speaking rate estimation (using zero crossings)
        zero_crossings = librosa.zero_crossings(audio)
        speech_rate = len(zero_crossings) / (len(audio) / self.sample_rate)
        
        # Voice quality measures
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        ).mean()
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        ).mean()
        
        return {
            **f0_stats,
            **energy_stats,
            'speech_rate': speech_rate,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff
        }
    
    def get_realtime_features(self, 
                            audio_chunk: np.ndarray,
                            prev_features: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Extract features suitable for real-time processing
        
        Args:
            audio_chunk: New audio chunk
            prev_features: Features from previous chunk for continuity
            
        Returns:
            Real-time compatible features
        """
        # Quick feature extraction for real-time
        f0, voiced_flag = librosa.pyin(
            audio_chunk,
            fmin=self.f0_min,
            fmax=self.f0_max,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # Frame energy (faster than full spectral computation)
        energy = librosa.feature.rms(
            y=audio_chunk,
            hop_length=self.hop_length
        )[0]
        
        # Basic spectral features
        mel_spec = librosa.feature.melspectrogram(
            y=audio_chunk,
            sr=self.sample_rate,
            n_mels=80,
            hop_length=self.hop_length
        )
        
        features = {
            'f0': f0,
            'voiced_flag': voiced_flag,
            'energy': energy,
            'mel_spectrogram': librosa.power_to_db(mel_spec)
        }
        
        # Ensure feature continuity if previous features provided
        if prev_features is not None:
            # Smooth discontinuities in f0
            if 'f0' in prev_features:
                overlap = min(len(features['f0']), 10)
                features['f0'][:overlap] = 0.5 * (
                    features['f0'][:overlap] + 
                    prev_features['f0'][-overlap:]
                )
            
            # Smooth energy transitions
            if 'energy' in prev_features:
                overlap = min(len(features['energy']), 10)
                features['energy'][:overlap] = 0.5 * (
                    features['energy'][:overlap] + 
                    prev_features['energy'][-overlap:]
                )
        
        return features