"""
Audio Processing Components for Emotion-Aware Voice Cloning
- Handles audio pre/post-processing
- Implements emotion-specific audio transformations
- Manages real-time audio processing pipeline
"""

import torch
import torchaudio
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import librosa
from scipy.signal import savgol_filter

class AudioProcessor:
    def __init__(self,
                 sample_rate: int = 22050,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: int = 1024,
                 n_mels: int = 80,
                 mel_fmin: float = 0.0,
                 mel_fmax: Optional[float] = None):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        
        # Initialize mel filterbank
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=mel_fmin,
            fmax=mel_fmax
        )
        self.mel_basis = torch.FloatTensor(self.mel_basis)
        
        # Initialize window function
        self.window = torch.hann_window(win_length)
        
    def process_audio(self,
                     waveform: torch.Tensor,
                     emotion_cond: Optional[torch.Tensor] = None
                     ) -> Dict[str, torch.Tensor]:
        """
        Process audio with emotion-aware transformations
        
        Args:
            waveform: Input waveform [1, T]
            emotion_cond: Optional emotion conditioning vector
            
        Returns:
            Dictionary of processed features
        """
        # Normalize audio
        waveform = self._normalize_audio(waveform)
        
        # Extract spectral features
        spec = self._stft(waveform)
        mel_spec = self._linear_to_mel(spec)
        
        # Extract prosodic features
        f0 = self._extract_f0(waveform)
        energy = self._extract_energy(spec)
        
        # Apply emotion-specific processing if provided
        if emotion_cond is not None:
            mel_spec = self._apply_emotion_transform(mel_spec, emotion_cond)
            f0 = self._modify_f0_for_emotion(f0, emotion_cond)
            energy = self._modify_energy_for_emotion(energy, emotion_cond)
        
        return {
            'waveform': waveform,
            'spectrogram': spec,
            'mel_spectrogram': mel_spec,
            'f0': f0,
            'energy': energy
        }
    
    def _normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1] range"""
        return waveform / (torch.max(torch.abs(waveform)) + 1e-8)
    
    def _stft(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute Short-time Fourier transform"""
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True
        )
        return torch.abs(stft)
    
    def _linear_to_mel(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Convert linear spectrogram to mel-scale"""
        return torch.matmul(self.mel_basis, spectrogram)
    
    def _extract_f0(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract F0 contour"""
        waveform_np = waveform.numpy()
        f0, voiced_flag, voiced_probs = librosa.pyin(
            waveform_np,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        
        # Interpolate unvoiced regions
        if np.sum(voiced_flag) > 0:
            f0[~voiced_flag] = np.interp(
                np.where(~voiced_flag)[0],
                np.where(voiced_flag)[0],
                f0[voiced_flag]
            )
        
        return torch.FloatTensor(f0)
    
    def _extract_energy(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Extract energy contour"""
        return torch.norm(spectrogram, dim=1)
    
    def _apply_emotion_transform(self,
                               mel_spec: torch.Tensor,
                               emotion_cond: torch.Tensor) -> torch.Tensor:
        """Apply emotion-specific spectral transformations"""
        # Extract emotion parameters
        valence, arousal, dominance = emotion_cond.split(1, dim=-1)
        
        # Spectral shaping based on emotion
        # High arousal -> enhance high frequencies
        if arousal > 0:
            high_freq_boost = torch.linspace(1.0, 1.0 + arousal.item(), self.n_mels)
            mel_spec = mel_spec * high_freq_boost.unsqueeze(1)
        
        # High valence -> smoother spectral envelope
        if valence > 0:
            mel_spec = torch.from_numpy(
                savgol_filter(mel_spec.numpy(), 5, 2, axis=0)
            )
            
        return mel_spec
    
    def _modify_f0_for_emotion(self,
                              f0: torch.Tensor,
                              emotion_cond: torch.Tensor) -> torch.Tensor:
        """Modify F0 contour based on emotion"""
        valence, arousal, dominance = emotion_cond.split(1, dim=-1)
        
        # Scale F0 based on arousal
        f0_scale = 1.0 + (0.2 * arousal.item())
        f0 = f0 * f0_scale
        
        # Add micro-prosody variations based on valence
        if valence > 0:
            variations = torch.sin(torch.linspace(0, 10*np.pi, len(f0)))
            variations = variations * (0.1 * valence.item())
            f0 = f0 + variations
            
        return f0
    
    def _modify_energy_for_emotion(self,
                                 energy: torch.Tensor,
                                 emotion_cond: torch.Tensor) -> torch.Tensor:
        """Modify energy contour based on emotion"""
        valence, arousal, dominance = emotion_cond.split(1, dim=-1)
        
        # Scale energy based on arousal and dominance
        energy_scale = 1.0 + (0.3 * arousal.item()) + (0.2 * dominance.item())
        energy = energy * energy_scale
        
        # Add dynamic variations based on valence
        if valence > 0:
            variations = torch.sin(torch.linspace(0, 5*np.pi, len(energy)))
            variations = variations * (0.15 * valence.item())
            energy = energy + variations
            
        return energy
    
    def inverse_transform(self,
                         mel_spec: torch.Tensor,
                         f0: Optional[torch.Tensor] = None,
                         emotion_cond: Optional[torch.Tensor] = None
                         ) -> torch.Tensor:
        """
        Convert mel-spectrogram back to waveform
        
        Args:
            mel_spec: Input mel-spectrogram
            f0: Optional F0 contour for synthesis
            emotion_cond: Optional emotion conditioning
            
        Returns:
            Synthesized waveform
        """
        # Convert mel to linear spectrogram
        spec = self._mel_to_linear(mel_spec)
        
        # Apply emotion-specific post-processing
        if emotion_cond is not None:
            spec = self._apply_emotion_postprocess(spec, emotion_cond)
            
        # Griffin-Lim algorithm for phase reconstruction
        angles = np.random.random_sample(spec.shape) * 2 * np.pi
        angles = np.cos(angles) + 1j * np.sin(angles)
        spec_complex = spec.numpy() * angles
        
        for i in range(50):
            inverse = librosa.istft(
                spec_complex,
                hop_length=self.hop_length,
                win_length=self.win_length
            )
            rebuilt = librosa.stft(
                inverse,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length
            )
            angles = rebuilt / np.abs(rebuilt)
            spec_complex = spec.numpy() * angles
            
        waveform = torch.FloatTensor(inverse)
        
        # Apply F0 modifications if provided
        if f0 is not None:
            waveform = self._apply_f0(waveform, f0)
            
        return waveform
    
    def _mel_to_linear(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Convert mel-spectrogram to linear spectrogram"""
        return torch.matmul(torch.pinverse(self.mel_basis), mel_spec)
    
    def _apply_emotion_postprocess(self,
                                 spec: torch.Tensor,
                                 emotion_cond: torch.Tensor) -> torch.Tensor:
        """Apply emotion-specific post-processing to spectrogram"""
        valence, arousal, dominance = emotion_cond.split(1, dim=-1)
        
        # Enhance spectral contrast for high arousal
        if arousal > 0:
            spec_db = librosa.amplitude_to_db(spec.numpy())
            spec_db = librosa.decompose.enhance(spec_db, rate=arousal.item())
            spec = torch.FloatTensor(librosa.db_to_amplitude(spec_db))
            
        return spec
    
    def _apply_f0(self, waveform: torch.Tensor, f0: torch.Tensor) -> torch.Tensor:
        """Apply F0 modifications to waveform"""
        # Implement pitch-shifting using phase vocoder
        stft = librosa.stft(
            waveform.numpy(),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        # Time-varying pitch shift
        modified = librosa.phase_vocoder(
            stft,
            f0 / torch.mean(f0),
            hop_length=self.hop_length
        )
        
        return torch.FloatTensor(librosa.istft(
            modified,
            hop_length=self.hop_length,
            win_length=self.win_length
        ))