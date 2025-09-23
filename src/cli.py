"""
Command Line Interface for Voice Cloning Tool
Compatible with Coqui TTS 0.27.1
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Import our modules
from voice_clone import VoiceClone
from processor import process_text
from audio_recorder import AudioRecorder
import soundfile as sf
import librosa
import numpy as np

# Import logging utilities from your setup
from utils.logging import get_logger, logger_manager

# Configure the logger for this script
log = get_logger('CLI')

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

def convert_to_wav(voice_path: str) -> str:
    """
    Fast WAV format check and conversion for XTTS-v2.
    Uses minimal processing for speed.
    """
    log.info(f"Checking audio file: {voice_path}")
    if not os.path.exists(voice_path):
        log.error(f"Audio file not found: {voice_path}")
        raise FileNotFoundError(f"Audio file not found: {voice_path}")
    
    # Quick metadata check for WAV files
    if voice_path.lower().endswith(".wav"):
        try:
            info = sf.info(voice_path)
            if info.samplerate == 22050 and info.channels == 1:
                log.info("Audio file is already in the correct format.")
                return voice_path
        except Exception:
            log.warning("Could not read WAV file metadata, attempting conversion.")

    # Create output path
    base_path = os.path.splitext(voice_path)[0]
    wav_path = f"{base_path}_converted.wav"
    
    # If file already exists and is fresh (less than 1 hour old), reuse it
    if os.path.exists(wav_path) and (time.time() - os.path.getmtime(wav_path)) < 3600:
        log.info(f"Using existing converted file: {wav_path}")
        return wav_path
    
    try:
        log.info("Converting audio format...")
        audio, sr = librosa.load(voice_path, sr=22050, mono=True, duration=30)
        
        log.info("Normalizing audio...")
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.95
        
        log.info("Saving converted audio...")
        sf.write(wav_path, audio, 22050)
        log.info(f"Converted audio saved to: {wav_path}")
        
        duration = len(audio) / sr
        log.info(f"Audio Info - Duration: {duration:.1f}s, Sample Rate: {sr}Hz")
        
        if duration < 6.0:
            log.warning("Voice sample is shorter than recommended 6 seconds. "
                        "Consider recording a longer sample for better quality.")
        
        return wav_path
        
    except Exception as e:
        log.error(f"Failed to convert audio: {e}", exc_info=True)
        raise ValueError(f"Failed to convert audio: {e}")

def record_voice_sample() -> str:
    """
    Record a voice sample using the microphone.
    Returns the path to the recorded WAV file.
    """
    log.info("Starting interactive voice recording session.")
    recorder = AudioRecorder(sample_rate=22050)
    
    # List available devices
    recorder.list_audio_devices()
    
    # Get device selection
    device_input = input("\nEnter device number (or press Enter for default): ").strip()
    device = None if not device_input.isdigit() else int(device_input)
    log.debug(f"Selected device: {device}")
    
    # Get recording duration
    duration_input = input("Recording duration in seconds (6+ recommended, default: 10): ").strip()
    duration = 10.0 if not duration_input else float(duration_input)
    log.debug(f"Selected duration: {duration} seconds")
    
    print("\nPress Enter when ready to start recording...")
    input()
    
    try:
        log.info(f"Recording starting in 3 seconds... will record for {duration:.1f}s")
        time.sleep(3)
        
        # Record audio
        audio_data, _ = recorder.record_voice_sample(duration, device)
        
        log.info("Recording completed.")
        
        # Show recording info
        info = recorder.get_audio_info(audio_data)
        log.info(f"Recording Information: Duration={info['duration']:.2f}s, Quality={'Good' if not info['is_silent'] else 'Too quiet'}")
        
        # Offer playback
        if input("\nPlay back recorded audio? (y/n): ").lower().strip() == 'y':
            log.info("Playing back recorded audio...")
            recorder.play_recorded_audio(audio_data)
        
        # Save to temporary file
        temp_path = recorder.get_temp_wav_path(audio_data)
        log.info(f"Recorded audio saved to temporary file: {temp_path}")
        
        return temp_path
        
    except Exception as e:
        log.error(f"Recording failed: {e}", exc_info=True)
        raise e

def validate_voice_sample(voice_wav: str, min_sec: float = 3.0, max_sec: float = 300.0):
    """
    Quick validation of voice sample using metadata.
    """
    try:
        info = sf.info(voice_wav)
        duration = info.duration
        log.debug(f"Validating voice sample duration: {duration:.1f}s")
        
        if duration < min_sec:
            log.warning(f"Voice sample is short ({duration:.1f}s). Minimum {min_sec}s recommended.")
        elif duration > max_sec:
            log.warning(f"Voice sample is very long ({duration:.1f}s). Consider trimming.")
        
        return True
        
    except Exception as e:
        log.error(f"Error validating audio: {e}", exc_info=True)
        return False

def main():
    logger_manager.start_operation('cli_execution')
    
    # Initialize voice cloner at startup for faster first generation
    log.info("Initializing VoiceClone singleton...")
    voice_cloner = VoiceClone()
    
    parser = argparse.ArgumentParser(
        description="Voice Cloning Tool using Coqui TTS 0.27.1 with XTTS-v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use an existing voice file:
  python cli.py --voice my_voice.wav --script script.txt --output speech.wav
  python cli.py --voice sample.mp3 --text "Hello world!" --output hello.wav --language en
  
  # Record voice using microphone:
  python cli.py --record --text "Hello world!" --output hello.wav
  python cli.py --record --script script.txt --output speech.wav --language en
        """
    )
    
    # Voice input options (either file or recording)
    voice_group = parser.add_mutually_exclusive_group(required=True)
    voice_group.add_argument(
        "--voice", "-v",
        help="Path to your voice sample (WAV/MP3/M4A, 6+ seconds recommended)"
    )
    voice_group.add_argument(
        "--record", "-r",
        action="store_true",
        help="Record voice sample using microphone"
    )
    
    # Text input (either --text or --script, but not both)
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument(
        "--text", "-t", 
        help="Text to convert to speech (use quotes for multiple words)"
    )
    text_group.add_argument(
        "--script", "-s", 
        help="Path to text file containing script to read"
    )
    
    parser.add_argument(
        "--output", "-o", 
        default="output.wav", 
        help="Output audio file path (default: output.wav)"
    )
    parser.add_argument(
        "--language", "-l", 
        default="en", 
        choices=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko", "hi"],
        help="Language code (default: en)"
    )
    
    # Voice style parameters
    parser.add_argument(
        "--emotion", "-e",
        type=float,
        default=1.0,
        help="Emotion intensity scale (0.5-1.5, default: 1.0)"
    )
    parser.add_argument(
        "--speed", "-sp",
        type=float,
        default=1.0,
        help="Speaking rate (0.5-2.0, default: 1.0)"
    )
    parser.add_argument(
        "--pitch", "-p",
        type=float,
        default=1.0,
        help="Voice pitch scale (0.5-2.0, default: 1.0)"
    )
    
    args = parser.parse_args()

    log.info("Voice Cloning Tool - Coqui TTS 0.27.1 Starting")
    log.info("=" * 50)
    
    try:
        # Step 1: Get voice sample (record or process file)
        if args.record:
            log.info("Using microphone for voice sample.")
            try:
                voice_wav = record_voice_sample()
            except Exception as e:
                log.error(f"Recording failed: {e}")
                return 1
        else:
            log.info("Using voice sample file.")
            voice_wav = convert_to_wav(args.voice)
        
        if not validate_voice_sample(voice_wav):
            log.error("Voice sample validation failed. Please check your audio file.")
            return 1

        # Step 2: Get text to synthesize
        if args.text:
            text = args.text
            log.info(f"Using provided text: {text[:50]}{'...' if len(text) > 50 else ''}")
        else:
            log.info(f"Reading script from: {args.script}")
            if not os.path.exists(args.script):
                log.error(f"Script file not found: {args.script}")
                return 1
                
            with open(args.script, "r", encoding="utf-8") as f:
                raw_text = f.read()
            text = process_text(raw_text)
            log.info(f"Script processed. Length: {len(text)} characters")

        # Step 3: Create output directory
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            log.debug(f"Output directory created: {output_dir}")

        # Step 4: Generate speech (voice_cloner already initialized)
        log.info("Generating speech in your cloned voice...")
        log.info(f"Language: {args.language}")
        
        # Validate style parameters
        args.emotion = max(0.5, min(1.5, args.emotion))
        args.speed = max(0.5, min(2.0, args.speed))
        args.pitch = max(0.5, min(2.0, args.pitch))
        
        log.info(f"Style Settings: Emotion={args.emotion:.1f}, Rate={args.speed:.1f}x, Pitch={args.pitch:.1f}")
        
        result_path = voice_cloner.generate(
            text=text,
            voice_sample=voice_wav,
            output_file=args.output,
            language=args.language,
            emotion_scale=args.emotion,
            speaking_rate=args.speed,
            pitch_scale=args.pitch
        )
        
        log.info("Voice cloning completed successfully!")
        log.info(f"Generated audio saved to: {result_path}")
        
        # Display final info
        try:
            audio, sr = sf.read(result_path)
            duration = len(audio) / sr
            log.info(f"Output audio: {duration:.2f} seconds, {sr} Hz")
        except Exception as e:
            log.warning(f"Could not read output file for final check: {e}")
            
        return 0
        
    except KeyboardInterrupt:
        log.info("\n\nVoice cloning interrupted by user")
        return 1
    except Exception as e:
        log.exception(f"An unexpected error occurred: {e}")
        return 1
    finally:
        logger_manager.end_operation('cli_execution')
        perf_metrics = logger_manager.get_performance_metrics('cli_execution')
        log.info(f"Total CLI execution time: {perf_metrics.get('average', 0):.2f} seconds")
        log.info("Exiting.")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)