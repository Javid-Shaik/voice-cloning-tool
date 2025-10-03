"""
Dataset class for emotion-aware voice cloning
"""
import os
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Optional

class VoiceCloneDataset(Dataset):
    """
    Dataset class for emotion-aware voice cloning training
    """
    def __init__(
        self,
        data_dir: str,
        config: Dict,
        split: str = "train",
        transform: Optional[callable] = None
    ):
        """
        Initialize the dataset
        
        Args:
            data_dir: Path to the data directory
            config: Configuration dictionary
            split: Dataset split ("train", "val", or "test")
            transform: Optional transform to apply to the data
        """
        super().__init__()
        self.data_dir = data_dir
        self.config = config
        self.split = split
        self.transform = transform
        
        # Audio parameters
        self.sample_rate = config['data']['sample_rate']
        self.n_fft = config['data']['n_fft']
        self.hop_length = config['data']['hop_length']
        self.win_length = config['data']['win_length']
        self.n_mels = config['data']['n_mels']
        
        # Data augmentation
        self.augment = config['data']['augmentation']['enabled'] and split == "train"
        if self.augment:
            self.pitch_shift_range = config['data']['augmentation']['pitch_shift']
            self.time_stretch_range = config['data']['augmentation']['time_stretch']
            self.volume_range = config['data']['augmentation']['volume']
        
        # Load data index
        self.data_index = self._build_data_index()
        
        # Initialize mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            power=2.0
        )
    
    def _build_data_index(self):
        """Build an index of all audio files and their metadata"""
        data_index = []
        split_dir = os.path.join(self.data_dir, self.split)
        
        for speaker_id in os.listdir(split_dir):
            speaker_dir = os.path.join(split_dir, speaker_id)
            if not os.path.isdir(speaker_dir):
                continue
            
            # Load speaker metadata if available
            metadata_path = os.path.join(speaker_dir, "metadata.txt")
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split("|")
                        if len(parts) >= 2:
                            audio_file, emotion = parts[:2]
                            metadata[audio_file] = {
                                "emotion": emotion,
                                "text": parts[2] if len(parts) > 2 else ""
                            }
            
            # Index audio files
            for audio_file in os.listdir(speaker_dir):
                if not audio_file.endswith((".wav", ".mp3")):
                    continue
                
                file_path = os.path.join(speaker_dir, audio_file)
                item = {
                    "audio_path": file_path,
                    "speaker_id": speaker_id
                }
                
                # Add metadata if available
                if audio_file in metadata:
                    item.update(metadata[audio_file])
                
                data_index.append(item)
        
        return data_index
    
    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file"""
        waveform, sr = torchaudio.load(audio_path)
        
        # Convert to mono if necessary
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        return waveform
    
    def _augment_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentations"""
        if not self.augment:
            return waveform
        
        # Pitch shift
        if np.random.random() < 0.5:
            n_steps = np.random.uniform(*self.pitch_shift_range)
            waveform = torchaudio.functional.pitch_shift(
                waveform, 
                self.sample_rate,
                n_steps
            )
        
        # Time stretch
        if np.random.random() < 0.5:
            rate = np.random.uniform(*self.time_stretch_range)
            waveform = torchaudio.functional.time_stretch(
                waveform,
                rate
            )
        
        # Volume adjustment
        if np.random.random() < 0.5:
            volume = np.random.uniform(*self.volume_range)
            waveform = waveform * volume
        
        return waveform
    
    def _extract_features(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract audio features"""
        # Compute mel spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec = torch.log1p(mel_spec)
        
        # Compute pitch (F0)
        f0 = torchaudio.functional.detect_pitch_frequency(
            waveform,
            self.sample_rate
        )
        
        # Extract prosody features (energy, duration)
        energy = torch.norm(waveform, dim=1)
        duration = waveform.size(1) / self.sample_rate
        
        return {
            "mel_spectrogram": mel_spec,
            "f0": f0,
            "energy": energy,
            "duration": torch.tensor([duration], dtype=torch.float32)
        }
    
    def _encode_emotion(self, emotion: str) -> torch.Tensor:
        """Encode emotion label as one-hot vector"""
        # TODO: Implement emotion encoding based on your emotion set
        emotion_map = {
            "neutral": 0,
            "happy": 1,
            "sad": 2,
            "angry": 3,
            "fear": 4
        }
        
        index = emotion_map.get(emotion.lower(), 0)
        one_hot = torch.zeros(len(emotion_map))
        one_hot[index] = 1.0
        return one_hot
    
    def __len__(self) -> int:
        return len(self.data_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data_index[idx]
        
        # Load and preprocess audio
        waveform = self._load_audio(item["audio_path"])
        
        # Apply augmentation for training
        if self.augment:
            waveform = self._augment_audio(waveform)
        
        # Extract features
        features = self._extract_features(waveform)
        
        # Add metadata
        features.update({
            "speaker_id": torch.tensor(int(item["speaker_id"])),
            "emotion": self._encode_emotion(item.get("emotion", "neutral")),
            "text": item.get("text", "")
        })
        
        # Apply additional transforms if specified
        if self.transform is not None:
            features = self.transform(features)
        
        return features