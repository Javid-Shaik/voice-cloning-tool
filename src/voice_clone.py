"""
Voice Cloner Class for Coqui TTS 0.27.1
Compatible with XTTS-v2 model
"""

import os
import torch
import time
from TTS.api import TTS
from progress import ProgressSpinner

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

class VoiceClone:
    def __init__(self):
        """
        Initialize TTS model with XTTS-v2 for proper voice cloning support.
        Compatible with coqui-tts==0.27.1
        """
        print("Initializing Voice Cloning with Coqui TTS 0.27.1...")
        
        # Use XTTS-v2 model which supports voice cloning
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        
        # Get device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize TTS with voice cloning support
        print("Loading XTTS-v2 model...")
        self.tts = TTS(model_name=model_name).to(self.device)
        print("XTTS-v2 model loaded successfully for voice cloning")

    def generate(self, text: str, voice_sample: str, output_file: str, language: str = "en"):
        """
        Generate speech in cloned voice using XTTS-v2.
        
        Args:
            text (str): Text to synthesize
            voice_sample (str): Path to reference audio file
            output_file (str): Path for output audio file
            language (str): Language code (en, es, fr, etc.)
        """
        # Validate inputs
        if not text.strip():
            raise ValueError("Text to synthesize is empty!")
            
        if not os.path.exists(voice_sample):
            raise FileNotFoundError(f"Voice sample not found: {voice_sample}")
            
        print(f"Cloning voice from: {voice_sample}")
        print(f"Text to synthesize: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"Output file: {output_file}")
        print(f"Language: {language}")

        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Show progress spinner during generation
            spinner = ProgressSpinner("Generating speech with cloned voice")
            spinner.start()
            
            try:
                # Generate speech with voice cloning
                # Note: In coqui-tts 0.27.1, we use speaker_wav parameter
                self.tts.tts_to_file(
                    text=text,
                    speaker_wav=voice_sample,  # Reference voice sample
                    file_path=output_file,
                    language=language,
                    split_sentences=True  # Better handling of long texts
                )
            finally:
                spinner.stop()
            
            print("\nVoice cloning completed successfully!")
            return output_file
            
        except Exception as e:
            print(f"Error during voice generation: {str(e)}")
            raise e
    
    def get_supported_languages(self):
        """Get list of supported languages"""
        return [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr", 
            "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko", "hi"
        ]

# Test function
if __name__ == "__main__":
    # Test the voice cloner
    cloner = VoiceClone()
    print("Voice cloner initialized successfully!")
    print(f"Supported languages: {', '.join(cloner.get_supported_languages())}")