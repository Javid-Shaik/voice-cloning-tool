"""kaggle_run.py
Helper script to run the voice-cloning generation on Kaggle GPU.
This script assumes the Kaggle kernel has GPU enabled and proper PyTorch/CUDA installed.
It is intentionally simple: loads the script and voice sample from the repo and runs
`VoiceClone.generate(...)` to produce an output file under /kaggle/working/output.
"""
import time
import os
from pathlib import Path

# Add repo root to path so imports work when executed from notebook
import sys
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.voice_clone import VoiceClone


def main():
    start = time.time()

    script_path = Path('data/script.txt')
    voice_sample = Path('data/voice_samples/my_voice.mp3')
    output_path = Path('/kaggle/working/output/speech_kaggle.wav')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    if not voice_sample.exists():
        raise FileNotFoundError(f"Voice sample not found: {voice_sample}")

    text = script_path.read_text(encoding='utf-8')

    cloner = VoiceClone()
    # Ensure model is loaded as early as possible
    try:
        result = cloner.generate(
            text=text,
            voice_sample=str(voice_sample),
            output_file=str(output_path),
            language='en',
            emotion_scale=1.0,
            speaking_rate=1.0,
            pitch_scale=1.0,
            quality_mode='ultra_fast'
        )
        elapsed = time.time() - start
        print(f"Generation finished in {elapsed:.2f}s. Output: {result}")
    except Exception as e:
        print(f"Generation failed: {e}")
        raise


if __name__ == '__main__':
    main()
