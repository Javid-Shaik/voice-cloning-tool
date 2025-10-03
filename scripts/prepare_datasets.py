"""
Dataset preparation script for emotion-aware voice cloning
Downloads and processes required datasets
"""

import os
import subprocess
import argparse
import urllib.request
import tarfile
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url: str, output_path: str, desc: str = "Downloading"):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_archive(archive_path: str, extract_path: str):
    """Extract tar.gz or zip archive"""
    logger.info(f"Extracting {archive_path}...")
    
    if archive_path.endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
    logger.info("Extraction complete")

def prepare_vctk(data_dir: Path):
    """Download and prepare VCTK dataset"""
    vctk_dir = data_dir / 'VCTK'
    vctk_dir.mkdir(parents=True, exist_ok=True)
    
    # Download VCTK
    vctk_url = "https://datashare.ed.ac.uk/download/DS_10283_3443.zip"
    archive_path = vctk_dir / "vctk.zip"
    
    if not archive_path.exists():
        logger.info("Downloading VCTK dataset...")
        download_url(vctk_url, str(archive_path), "Downloading VCTK")
        extract_archive(str(archive_path), str(vctk_dir))
        archive_path.unlink()  # Remove archive after extraction

def prepare_emov_db(data_dir: Path):
    """Download and prepare EmoV-DB dataset"""
    emov_dir = data_dir / 'EmoV-DB'
    emov_dir.mkdir(parents=True, exist_ok=True)
    
    # Download EmoV-DB (requires manual download due to license)
    logger.info("EmoV-DB requires manual download from: https://emotional-voices.com/")
    logger.info(f"Please download and extract the dataset to: {emov_dir}")

def prepare_libritts(data_dir: Path):
    """Download and prepare LibriTTS dataset"""
    libritts_dir = data_dir / 'LibriTTS'
    libritts_dir.mkdir(parents=True, exist_ok=True)
    
    # Download LibriTTS
    libritts_url = "http://www.openslr.org/resources/60/train-clean-100.tar.gz"
    archive_path = libritts_dir / "libritts.tar.gz"
    
    if not archive_path.exists():
        logger.info("Downloading LibriTTS dataset...")
        download_url(libritts_url, str(archive_path), "Downloading LibriTTS")
        extract_archive(str(archive_path), str(libritts_dir))
        archive_path.unlink()  # Remove archive after extraction

def process_datasets(data_dir: Path):
    """Process all datasets into a unified format"""
    logger.info("Processing datasets...")
    
    # Create processed directory
    processed_dir = data_dir / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each dataset
    datasets = ['VCTK', 'EmoV-DB', 'LibriTTS']
    for dataset in datasets:
        src_dir = data_dir / dataset
        if src_dir.exists():
            logger.info(f"Processing {dataset}...")
            
            # Create dataset-specific directory
            dst_dir = processed_dir / dataset.lower()
            dst_dir.mkdir(exist_ok=True)
            
            # Copy and organize files
            for audio_file in src_dir.rglob('*.wav'):
                # Create relative path structure
                rel_path = audio_file.relative_to(src_dir)
                dst_path = dst_dir / rel_path
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(audio_file, dst_path)
        else:
            logger.warning(f"{dataset} directory not found, skipping...")
    
    logger.info("Dataset processing complete")

def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for emotion-aware voice cloning")
    parser.add_argument('--data_dir', type=str, default='data',
                      help='Directory to store datasets')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare each dataset
    prepare_vctk(data_dir)
    prepare_emov_db(data_dir)
    prepare_libritts(data_dir)
    
    # Process datasets
    process_datasets(data_dir)
    
    logger.info("All datasets prepared successfully!")

if __name__ == "__main__":
    main()