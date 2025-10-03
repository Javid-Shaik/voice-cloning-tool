"""
Feature extraction script for preprocessing datasets
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor
import json

from features.linguistic_features import LinguisticFeatureExtractor
from features.emotion_features import EmotionFeatureExtractor
from features.acoustic_features import AcousticFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_features_for_file(args):
    """Extract features for a single audio file"""
    audio_path, text, emotion, feature_extractors = args
    
    try:
        # Extract linguistic features
        linguistic_features = feature_extractors['linguistic'].extract_features(text)
        
        # Extract emotion features
        emotion_features = feature_extractors['emotion'].extract_features(
            text,
            emotion=emotion
        )
        
        # Extract acoustic features
        acoustic_features = feature_extractors['acoustic'].extract_features(
            audio_path
        )
        
        return {
            'audio_path': str(audio_path),
            'features': {
                'linguistic': linguistic_features,
                'emotion': emotion_features,
                'acoustic': acoustic_features
            }
        }
    except Exception as e:
        logger.error(f"Failed to process {audio_path}: {e}")
        return None

def process_dataset(dataset_dir: Path, output_dir: Path, num_workers: int = 4):
    """Process entire dataset"""
    logger.info(f"Processing dataset in {dataset_dir}")
    
    # Initialize feature extractors
    feature_extractors = {
        'linguistic': LinguisticFeatureExtractor(),
        'emotion': EmotionFeatureExtractor(),
        'acoustic': AcousticFeatureExtractor()
    }
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    features_dir = output_dir / 'features'
    features_dir.mkdir(exist_ok=True)
    
    # Load metadata if exists
    metadata_path = dataset_dir / 'metadata.json'
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        return
        
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Prepare processing arguments
    process_args = []
    for item in metadata:
        audio_path = dataset_dir / item['audio_file']
        if audio_path.exists():
            process_args.append((
                audio_path,
                item.get('text', ''),
                item.get('emotion', 'neutral'),
                feature_extractors
            ))
    
    # Process files in parallel
    logger.info(f"Processing {len(process_args)} files with {num_workers} workers")
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(
            executor.map(extract_features_for_file, process_args),
            total=len(process_args),
            desc="Extracting features"
        ):
            if result is not None:
                results.append(result)
    
    # Save features
    logger.info(f"Saving {len(results)} processed files")
    for result in results:
        audio_path = Path(result['audio_path'])
        output_path = features_dir / f"{audio_path.stem}.pt"
        
        # Convert features to tensors
        features = {
            k: {
                k2: torch.tensor(v2) if isinstance(v2, np.ndarray) else v2
                for k2, v2 in v.items()
            }
            for k, v in result['features'].items()
        }
        
        # Save features
        torch.save(features, output_path)
    
    logger.info("Feature extraction complete!")

def main():
    parser = argparse.ArgumentParser(description="Extract features from datasets")
    parser.add_argument('--data_dir', type=str, default='data/processed',
                      help='Directory containing processed datasets')
    parser.add_argument('--output_dir', type=str, default='data/features',
                      help='Directory to save extracted features')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of parallel workers')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    # Process each dataset
    for dataset_dir in data_dir.iterdir():
        if dataset_dir.is_dir():
            logger.info(f"Processing dataset: {dataset_dir.name}")
            process_dataset(
                dataset_dir,
                output_dir / dataset_dir.name,
                args.num_workers
            )

if __name__ == "__main__":
    main()