"""
Audio Recording Module for Voice Cloning
Records audio from microphone without saving to disk
"""


import sounddevice as sd
import numpy as np
import io
import soundfile as sf
import tempfile
import os
import queue
import threading
from typing import Tuple, Union
from progress import RecordingProgress


class AudioRecorder:
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize audio recorder
        
        Args:
            sample_rate (int): Audio sample rate for recording
        """
        self.sample_rate = sample_rate
        self.audio_data = None
        
    def list_audio_devices(self):
        """List available audio input devices"""
        print("Available audio input devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  [{i}] {device['name']} - {device['max_input_channels']} channels")
    
    def record_voice_sample(self, duration: float = 10.0, device=None) -> Tuple[np.ndarray, int]:
        """
        Record audio from microphone into memory
        
        Args:
            duration (float): Recording duration in seconds
            device: Audio device index (None for default)
            
        Returns:
            Tuple[np.ndarray, int]: Audio data and sample rate
        """
        print("Speak clearly into your microphone...")
        
        try:
            # Create queue for collecting audio chunks
            q = queue.Queue()
            audio_data = np.zeros((int(duration * self.sample_rate),), dtype='float32')
            current_position = 0
            
            def audio_callback(indata, frames, time, status):
                nonlocal current_position
                if status:
                    print(f"Status: {status}")
                chunk = indata.copy()
                q.put(chunk)
                end_pos = min(current_position + len(chunk), len(audio_data))
                audio_data[current_position:end_pos] = chunk[:end_pos-current_position].flatten()
                current_position = end_pos
            
            # Create progress display
            progress = RecordingProgress(duration)
            
            # Start recording with callback
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32',
                callback=audio_callback,
                device=device
            )
            
            with stream:
                progress.start(lambda: audio_data)
                sd.sleep(int(duration * 1000))
                progress.stop()
            
            # Process recorded audio
            while not q.empty():
                chunk = q.get()
                if current_position < len(audio_data):
                    end_pos = min(current_position + len(chunk), len(audio_data))
                    audio_data[current_position:end_pos] = chunk[:end_pos-current_position].flatten()
                    current_position = end_pos
            
            # Basic validation
            if np.max(np.abs(audio_data)) < 0.01:
                print("\nWarning: Recorded audio is very quiet. Check your microphone.")
            
            self.audio_data = audio_data
            return audio_data, self.sample_rate
            
        except Exception as e:
            print(f"Error during recording: {e}")
            raise e
    
    def get_temp_wav_path(self, audio_data: np.ndarray = None) -> str:
        """
        Create a temporary WAV file from recorded audio
        File will be automatically deleted when the program exits
        
        Args:
            audio_data: Audio data array (uses last recorded if None)
            
        Returns:
            str: Path to temporary WAV file
        """
        if audio_data is None:
            if self.audio_data is None:
                raise ValueError("No audio data available. Record audio first.")
            audio_data = self.audio_data
        
        # Create temporary file that will be deleted automatically
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Write audio data to temporary file
        sf.write(temp_path, audio_data, self.sample_rate)
        
        return temp_path
    
    def audio_to_bytes_io(self, audio_data: np.ndarray = None) -> io.BytesIO:
        """
        Convert audio data to BytesIO object (in-memory WAV)
        
        Args:
            audio_data: Audio data array (uses last recorded if None)
            
        Returns:
            io.BytesIO: In-memory WAV file
        """
        if audio_data is None:
            if self.audio_data is None:
                raise ValueError("No audio data available. Record audio first.")
            audio_data = self.audio_data
        
        # Create in-memory WAV file
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, self.sample_rate, format='WAV')
        buffer.seek(0)  # Reset to beginning
        
        return buffer
    
    def play_recorded_audio(self, audio_data: np.ndarray = None):
        """
        Play back the recorded audio for verification
        
        Args:
            audio_data: Audio data to play (uses last recorded if None)
        """
        if audio_data is None:
            if self.audio_data is None:
                print("No audio data to play. Record audio first.")
                return
            audio_data = self.audio_data
        
        print("Playing back recorded audio...")
        sd.play(audio_data, self.sample_rate)
        sd.wait()
        print("Playback finished")
    
    def get_audio_info(self, audio_data: np.ndarray = None) -> dict:
        """
        Get information about the recorded audio
        
        Args:
            audio_data: Audio data to analyze (uses last recorded if None)
            
        Returns:
            dict: Audio information
        """
        if audio_data is None:
            if self.audio_data is None:
                return {"error": "No audio data available"}
            audio_data = self.audio_data
        
        duration = len(audio_data) / self.sample_rate
        max_amplitude = np.max(np.abs(audio_data))
        rms_level = np.sqrt(np.mean(audio_data ** 2))
        
        return {
            "duration": duration,
            "sample_rate": self.sample_rate,
            "max_amplitude": max_amplitude,
            "rms_level": rms_level,
            "samples": len(audio_data),
            "is_silent": max_amplitude < 0.01
        }


def interactive_recording_session():
    """
    Interactive session for recording voice samples
    """
    recorder = AudioRecorder()
    
    print("Voice Recording Session")
    print("=" * 30)
    
    # List available devices
    print("\nStep 1: Audio Devices")
    recorder.list_audio_devices()
    
    # Get device selection
    device_input = input("\nEnter device number (or press Enter for default): ").strip()
    device = None
    if device_input.isdigit():
        device = int(device_input)
    
    # Get recording duration
    duration_input = input("Recording duration in seconds (default: 10): ").strip()
    duration = 10.0
    if duration_input:
        try:
            duration = float(duration_input)
        except ValueError:
            print("Invalid duration, using default: 10 seconds")
    
    # Record audio
    print(f"\nStep 2: Recording")
    print("Press Enter when ready to start recording...")
    input()
    
    try:
        audio_data, sample_rate = recorder.record_voice_sample(duration, device)
        
        # Show audio info
        info = recorder.get_audio_info(audio_data)
        print(f"\nRecording Information:")
        print(f"  Duration: {info['duration']:.2f} seconds")
        print(f"  Sample Rate: {info['sample_rate']} Hz")
        print(f"  Max Amplitude: {info['max_amplitude']:.3f}")
        print(f"  Quality: {'Good' if not info['is_silent'] else 'Too quiet'}")
        
        # Playback option
        playback = input("\nPlay back recorded audio? (y/n): ").lower().strip()
        if playback == 'y':
            recorder.play_recorded_audio(audio_data)
        
        return recorder, audio_data
        
    except Exception as e:
        print(f"Recording failed: {e}")
        return None, None


if __name__ == "__main__":
    # Test the audio recording
    recorder, audio_data = interactive_recording_session()
    
    if audio_data is not None:
        print("\nRecording successful!")
        print("You can now use this audio data for voice cloning.")
    else:
        print("Recording failed or cancelled.")