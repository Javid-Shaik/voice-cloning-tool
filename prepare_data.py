"""
Script to prepare training data from raw audio files
"""
import argparse
import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import librosa
import pandas as pd
from src.voice_clone import UltimateVoiceClone
from src.utils.logging import get_logger

log = get_logger('prepare_data')

def extract_features(file_path, config):
    """Extract features from audio file"""
    try:
        # Load audio
        waveform, sr = torchaudio.load(file_path)
        
        # Convert to mono if needed
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sr != config['data']['sample_rate']:
            resampler = torchaudio.transforms.Resample(sr, config['data']['sample_rate'])
            waveform = resampler(waveform)
        
        # Extract mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config['data']['sample_rate'],
            n_fft=config['data']['n_fft'],
            hop_length=config['data']['hop_length'],
            win_length=config['data']['win_length'],
            n_mels=config['data']['n_mels']
        )
        mel_spec = mel_transform(waveform)
        mel_spec = torch.log1p(mel_spec)
        
        # Extract F0
        f0 = librosa.yin(
            waveform.numpy().squeeze(),
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=config['data']['sample_rate'],
            hop_length=config['data']['hop_length']
        )
        
        # Extract energy
        energy = torch.norm(waveform, dim=1)
        
        # Extract duration
        duration = len(waveform[0]) / config['data']['sample_rate']
        
        return {
            'mel_spectrogram': mel_spec.numpy(),
            'f0': f0,
            'energy': energy.numpy(),
            'duration': duration,
            'sample_rate': config['data']['sample_rate']
        }
        
    except Exception as e:
        log.error(f"Feature extraction failed for {file_path}: {e}")
        return None

def process_emotions_file(emotions_file):
    """Process emotions annotation file"""
    try:
        df = pd.read_csv(emotions_file, header=None)
        emotions = {}
        
        for _, row in df.iterrows():
            # Expected format: filename,emotion,intensity
            if len(row) >= 2:
                filename = row[0]
                emotion = row[1].lower()
                intensity = float(row[2]) if len(row) > 2 else 1.0
                emotions[filename] = {
                    'emotion': emotion,
                    'intensity': intensity
                }
        return emotions
    except Exception as e:
        log.error(f"Failed to process emotions file {emotions_file}: {e}")
        return {}

def prepare_dataset(args, config):
    """Prepare dataset from raw audio files"""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    emotions_file = args.emotions_file
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Load emotions if available
    emotions = {}
    if emotions_file and os.path.exists(emotions_file):
        emotions = process_emotions_file(emotions_file)
    
    # Collect audio files
    audio_files = []
    for ext in ['.wav', '.mp3']:
        audio_files.extend(input_dir.rglob(f'*{ext}'))
    
    if not audio_files:
        raise ValueError(f"No audio files found in {input_dir}")
    
    log.info(f"Found {len(audio_files)} audio files")
    
    # Split files
    np.random.seed(config['environment']['seed'])
    np.random.shuffle(audio_files)
    
    total = len(audio_files)
    train_idx = int(total * 0.8)
    val_idx = int(total * 0.9)
    
    splits = {
        'train': audio_files[:train_idx],
        'val': audio_files[train_idx:val_idx],
        'test': audio_files[val_idx:]
    }
    
    # Process files
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for split, files in splits.items():
            log.info(f"Processing {split} split ({len(files)} files)")
            
            futures = []
            for file in files:
                extract_fn = partial(extract_features, config=config)
                futures.append((file, executor.submit(extract_fn, str(file))))
            
            for file, future in tqdm(futures, desc=f"Processing {split}"):
                try:
                    features = future.result()
                    if features is None:
                        continue
                    
                    # Get relative path to preserve structure
                    rel_path = file.relative_to(input_dir)
                    out_path = output_dir / split / rel_path
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save features
                    np.savez(
                        out_path.with_suffix('.npz'),
                        **features
                    )
                    
                    # Save metadata
                    metadata = {
                        'features_file': str(out_path.with_suffix('.npz')),
                        'original_file': str(file),
                        'split': split
                    }
                    
                    # Add emotion if available
                    if file.name in emotions:
                        metadata.update(emotions[file.name])
                    
                    with open(out_path.with_suffix('.json'), 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                except Exception as e:
                    log.error(f"Failed to process {file}: {e}")
                    continue
    
    log.info("Dataset preparation completed")

def main():
    parser = argparse.ArgumentParser(description="Prepare training dataset")
    parser.add_argument('--input-dir', type=str, required=True,
                       help="Directory containing raw audio files")
    parser.add_argument('--output-dir', type=str, required=True,
                       help="Directory to save processed features")
    parser.add_argument('--emotions-file', type=str,
                       help="CSV file mapping filenames to emotions")
    parser.add_argument('--config', type=str, default="configs/training.yaml",
                       help="Training configuration file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    prepare_dataset(args, config)

if __name__ == "__main__":
    main()