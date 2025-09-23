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

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

def convert_to_wav(voice_path: str) -> str:
    """
    Convert audio to proper format for XTTS-v2 (22050Hz, mono, WAV).
    Compatible with coqui-tts 0.27.1
    """
    print(f"Checking audio file: {voice_path}")
    
    if not os.path.exists(voice_path):
        raise FileNotFoundError(f"Audio file not found: {voice_path}")
    
    # If already WAV, check if format is correct
    if voice_path.lower().endswith(".wav"):
        try:
            audio, sr = sf.read(voice_path)
            # Check if it's mono and correct sample rate
            is_mono = len(audio.shape) == 1 or (len(audio.shape) == 2 and audio.shape[1] == 1)
            if sr == 22050 and is_mono:
                print("Audio file already in correct format")
                return voice_path
        except Exception as e:
            print(f"Error reading WAV file: {e}")

    print("Converting audio to proper format (22050Hz, mono, WAV)...")
    
    # Create output path
    base_path = os.path.splitext(voice_path)[0]
    wav_path = f"{base_path}_converted.wav"
    
    try:
        # Load audio with librosa (handles most formats)
        audio, sr = librosa.load(voice_path, sr=22050, mono=True)
        
        # Normalize audio to prevent clipping
        audio = librosa.util.normalize(audio)
        
        # Save as WAV
        sf.write(wav_path, audio, 22050)
        print(f"Audio converted and saved to: {wav_path}")
        
        return wav_path
        
    except Exception as e:
        raise ValueError(f"Failed to convert audio: {e}")

def record_voice_sample() -> str:
    """
    Record a voice sample using the microphone.
    Returns the path to the recorded WAV file.
    """
    print("\nRecording Voice Sample")
    print("=" * 50)
    
    recorder = AudioRecorder(sample_rate=22050)
    
    # List available devices
    recorder.list_audio_devices()
    
    # Get device selection
    device_input = input("\nEnter device number (or press Enter for default): ").strip()
    device = None if not device_input.isdigit() else int(device_input)
    
    # Get recording duration
    duration_input = input("Recording duration in seconds (6+ recommended, default: 10): ").strip()
    duration = 10.0 if not duration_input else float(duration_input)
    
    print("\nPress Enter when ready to start recording...")
    input()
    
    try:
        print("\n3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("\nRecording started - will stop automatically after {:.1f} seconds".format(duration))
        print("=" * 50)
        
        # Record audio
        audio_data, _ = recorder.record_voice_sample(duration, device)
        
        print("\nRecording completed!")
        
        # Show recording info
        info = recorder.get_audio_info(audio_data)
        print(f"\nRecording Information:")
        print(f"   Duration: {info['duration']:.2f} seconds")
        print(f"   Quality: {'Good' if not info['is_silent'] else 'Too quiet'}")
        
        # Offer playback
        if input("\nPlay back recorded audio? (y/n): ").lower().strip() == 'y':
            recorder.play_recorded_audio(audio_data)
        
        # Save to temporary file
        temp_path = recorder.get_temp_wav_path(audio_data)
        print(f"\nRecorded audio saved to: {temp_path}")
        
        return temp_path
        
    except Exception as e:
        print(f"Recording failed: {e}")
        raise e

def validate_voice_sample(voice_wav: str, min_sec: float = 3.0, max_sec: float = 300.0):
    """
    Validate voice sample for XTTS-v2 compatibility.
    """
    try:
        audio, sr = sf.read(voice_wav)
        duration = len(audio) / sr
        
        print(f"Voice sample analysis:")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Sample rate: {sr} Hz")
        print(f"   Channels: {1 if len(audio.shape) == 1 else audio.shape[1]}")
        
        if duration < min_sec:
            print(f"Warning: Voice sample is short ({duration:.2f}s). Minimum {min_sec}s recommended.")
            print("   Consider using a longer sample for better voice cloning quality.")
        elif duration < 6.0:
            print(f"Note: Voice sample is {duration:.2f}s. 6+ seconds optimal for XTTS-v2.")
        elif duration > max_sec:
            print(f"Warning: Voice sample is very long ({duration:.2f}s). Consider trimming.")
        else:
            print(f"Voice sample duration is good for voice cloning")
            
        return True
        
    except Exception as e:
        print(f"Error validating audio: {e}")
        return False

def main():
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
    
    args = parser.parse_args()

    print("Voice Cloning Tool - Coqui TTS 0.27.1")
    print("=" * 50)
    
    try:
        # Step 1: Get voice sample (record or process file)
        if args.record:
            print("Starting voice recording session...")
            try:
                voice_wav = record_voice_sample()
            except Exception as e:
                print(f"Recording failed: {e}")
                return 1
        else:
            print("Processing voice sample...")
            voice_wav = convert_to_wav(args.voice)
        
        if not validate_voice_sample(voice_wav):
            print("Voice sample validation failed. Please check your audio file.")
            return 1

        # Step 2: Get text to synthesize
        if args.text:
            text = args.text
            print(f"Using provided text: {text[:50]}{'...' if len(text) > 50 else ''}")
        else:
            print(f"Reading script from: {args.script}")
            if not os.path.exists(args.script):
                print(f"Script file not found: {args.script}")
                return 1
                
            with open(args.script, "r", encoding="utf-8") as f:
                raw_text = f.read()
            text = process_text(raw_text)
            print(f"Script processed. Length: {len(text)} characters")

        # Step 3: Create output directory
        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output directory: {output_dir}")

        # Step 4: Initialize voice cloner
        print("\nInitializing voice cloning...")
        voice_cloner = VoiceClone()
        
        # Step 5: Generate speech
        print(f"\nGenerating speech in your cloned voice...")
        print(f"Language: {args.language}")
        
        result_path = voice_cloner.generate(
            text=text,
            voice_sample=voice_wav,
            output_file=args.output,
            language=args.language
        )

        print(f"Generated audio saved to: {result_path}")
        
        # Display final info
        try:
            audio, sr = sf.read(result_path)
            duration = len(audio) / sr
            print(f"Output audio: {duration:.2f} seconds, {sr} Hz")
        except:
            pass
            
        return 0
        
    except KeyboardInterrupt:
        print("\n\nVoice cloning interrupted by user")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)