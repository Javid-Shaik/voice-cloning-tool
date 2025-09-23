#!/usr/bin/env python3
"""
Quick Test Script for Coqui TTS 0.27.1 Voice Cloning
Run this first to test your installation
"""

import os
import torch
from TTS.api import TTS

def test_installation():
    """Test if TTS is working properly"""
    print("Testing Coqui TTS 0.27.1 Installation...")
    
    try:
        # Check device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")
        
        # Load model
        print("Loading XTTS-v2 model...")
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print("Model loaded successfully!")
        
        # Check supported languages
        print(f"Supported languages: {len(tts.languages)} languages")
        print(f"Languages: {', '.join(tts.languages[:10])}...")
        
        print("\nInstallation test PASSED!")
        print("Your voice cloning setup is ready to use!")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Try: pip install coqui-tts==0.27.1")
        return False
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def create_sample_script():
    """Create a sample text for voice cloning"""
    sample_text = """
    Hello! Welcome to my voice cloning demonstration.
    This is an amazing technology that can clone any voice using just a few seconds of audio.
    I can now read any text in my own voice, which is pretty incredible.
    Python and artificial intelligence make this possible, and it runs completely offline.
    Thank you for trying out this voice cloning tool!
    """
    
    with open("sample_script.txt", "w", encoding="utf-8") as f:
        f.write(sample_text.strip())
    
    print("Created sample_script.txt for testing")

if __name__ == "__main__":
    print("=" * 50)
    print("Coqui TTS 0.27.1 - Voice Cloning Test")
    print("=" * 50)
    
    # Test installation
    if test_installation():
        create_sample_script()
        
        print("\nNext Steps:")
        print("1. Record 6-30 seconds of your voice and save as 'my_voice.wav'")
        print("2. Run: python voice_clone.py")
        print("3. Your cloned voice will be saved as 'cloned_speech.wav'")
        
        print("\nQuick Test Command:")
        print("python -c \"from TTS.api import TTS; print('TTS ready!')\"")
    else:
        print("\nInstallation test failed. Please check your setup.")