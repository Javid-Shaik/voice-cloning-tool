"""
Quality Assurance System for Voice Cloning
- Implements metrics for voice quality assessment
- Handles emotion similarity evaluation
- Provides prosody accuracy metrics
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import librosa
from scipy.stats import pearsonr
from pesq import pesq
from pystoi import stoi

class VoiceQualityMetrics:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        
    def compute_all_metrics(self,
                          generated: torch.Tensor,
                          reference: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive voice quality metrics
        
        Args:
            generated: Generated audio waveform
            reference: Reference audio waveform
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Convert to numpy if needed
        if torch.is_tensor(generated):
            generated = generated.numpy()
        if torch.is_tensor(reference):
            reference = reference.numpy()
            
        # Basic audio metrics
        metrics.update(self._compute_audio_metrics(generated, reference))
        
        # Prosody metrics
        metrics.update(self._compute_prosody_metrics(generated, reference))
        
        # Perceptual metrics
        metrics.update(self._compute_perceptual_metrics(generated, reference))
        
        return metrics
    
    def _compute_audio_metrics(self,
                             generated: np.ndarray,
                             reference: np.ndarray) -> Dict[str, float]:
        """Compute basic audio quality metrics"""
        metrics = {}
        
        # Signal-to-Noise Ratio (SNR)
        noise = generated - reference
        signal_power = np.mean(reference ** 2)
        noise_power = np.mean(noise ** 2)
        metrics['snr'] = 10 * np.log10(signal_power / noise_power)
        
        # Root Mean Square Error (RMSE)
        metrics['rmse'] = np.sqrt(np.mean((generated - reference) ** 2))
        
        # PESQ (Perceptual Evaluation of Speech Quality)
        try:
            metrics['pesq'] = pesq(
                self.sample_rate, reference, generated, 'wb'
            )
        except:
            metrics['pesq'] = 0.0
            
        # STOI (Short-Time Objective Intelligibility)
        try:
            metrics['stoi'] = stoi(
                reference, generated, self.sample_rate, extended=False
            )
        except:
            metrics['stoi'] = 0.0
            
        return metrics
    
    def _compute_prosody_metrics(self,
                               generated: np.ndarray,
                               reference: np.ndarray) -> Dict[str, float]:
        """Compute prosody similarity metrics"""
        metrics = {}
        
        # Extract F0 contours
        f0_gen = self._extract_f0(generated)
        f0_ref = self._extract_f0(reference)
        
        # F0 correlation
        if len(f0_gen) == len(f0_ref):
            corr, _ = pearsonr(f0_gen, f0_ref)
            metrics['f0_correlation'] = corr
        else:
            metrics['f0_correlation'] = 0.0
            
        # F0 RMSE
        f0_rmse = np.sqrt(np.mean((f0_gen - f0_ref) ** 2))
        metrics['f0_rmse'] = f0_rmse
        
        # Energy contour similarity
        energy_gen = np.abs(librosa.stft(generated))
        energy_ref = np.abs(librosa.stft(reference))
        
        energy_corr = np.corrcoef(
            np.mean(energy_gen, axis=0),
            np.mean(energy_ref, axis=0)
        )[0,1]
        metrics['energy_correlation'] = energy_corr
        
        return metrics
    
    def _compute_perceptual_metrics(self,
                                  generated: np.ndarray,
                                  reference: np.ndarray) -> Dict[str, float]:
        """Compute perceptual quality metrics"""
        metrics = {}
        
        # Mel-cepstral distortion (MCD)
        mfcc_gen = librosa.feature.mfcc(y=generated, sr=self.sample_rate)
        mfcc_ref = librosa.feature.mfcc(y=reference, sr=self.sample_rate)
        
        if mfcc_gen.shape == mfcc_ref.shape:
            mcd = np.mean(np.sqrt(np.sum((mfcc_gen - mfcc_ref) ** 2, axis=0)))
            metrics['mcd'] = mcd
        else:
            metrics['mcd'] = float('inf')
            
        # Spectral convergence
        spec_gen = np.abs(librosa.stft(generated))
        spec_ref = np.abs(librosa.stft(reference))
        
        if spec_gen.shape == spec_ref.shape:
            num = np.linalg.norm(spec_gen - spec_ref)
            den = np.linalg.norm(spec_ref)
            metrics['spectral_convergence'] = num / (den + 1e-8)
        else:
            metrics['spectral_convergence'] = float('inf')
            
        return metrics
        
    def _extract_f0(self, audio: np.ndarray) -> np.ndarray:
        """Extract F0 contour using librosa"""
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        return f0[voiced_flag]

class EmotionSimilarityMetrics:
    def __init__(self, emotion_model):
        self.emotion_model = emotion_model
        
    def compute_emotion_similarity(self,
                                 generated: torch.Tensor,
                                 reference: torch.Tensor) -> Dict[str, float]:
        """
        Compute emotion similarity metrics
        
        Args:
            generated: Generated speech features
            reference: Reference speech features
            
        Returns:
            Dictionary of emotion similarity metrics
        """
        # Extract emotion embeddings
        gen_emotion = self.emotion_model.extract_features(generated)
        ref_emotion = self.emotion_model.extract_features(reference)
        
        metrics = {}
        
        # Emotion classification accuracy
        gen_probs = gen_emotion['emotion_probs']
        ref_probs = ref_emotion['emotion_probs']
        
        metrics['emotion_kl_div'] = torch.nn.functional.kl_div(
            gen_probs.log(), ref_probs
        ).item()
        
        # VAD space similarity
        gen_vad = gen_emotion['emotion_vad']
        ref_vad = ref_emotion['emotion_vad']
        
        metrics['vad_cosine'] = torch.nn.functional.cosine_similarity(
            gen_vad, ref_vad
        ).item()
        
        metrics['vad_l2'] = torch.norm(gen_vad - ref_vad).item()
        
        return metrics

class QualityAssurance:
    def __init__(self,
                 sample_rate: int = 22050,
                 emotion_model = None):
        self.voice_metrics = VoiceQualityMetrics(sample_rate)
        self.emotion_metrics = EmotionSimilarityMetrics(emotion_model)
        
    def evaluate_synthesis(self,
                         generated: torch.Tensor,
                         reference: torch.Tensor) -> Dict[str, float]:
        """
        Comprehensive synthesis quality evaluation
        
        Args:
            generated: Generated speech
            reference: Reference speech
            
        Returns:
            Dictionary of all quality metrics
        """
        # Get voice quality metrics
        metrics = self.voice_metrics.compute_all_metrics(
            generated, reference
        )
        
        # Get emotion similarity metrics
        emotion_metrics = self.emotion_metrics.compute_emotion_similarity(
            generated, reference
        )
        metrics.update(emotion_metrics)
        
        # Add overall quality score
        metrics['overall_quality'] = self._compute_overall_score(metrics)
        
        return metrics
    
    def _compute_overall_score(self, metrics: Dict[str, float]) -> float:
        """Compute weighted overall quality score"""
        weights = {
            'pesq': 0.3,
            'stoi': 0.2,
            'f0_correlation': 0.15,
            'energy_correlation': 0.15,
            'emotion_kl_div': 0.1,
            'vad_cosine': 0.1
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                value = metrics[metric]
                if metric == 'emotion_kl_div':
                    value = 1.0 / (1.0 + value)  # Convert to similarity
                score += weight * value
                
        return score