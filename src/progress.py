"""
Progress indicators for CLI interface
"""
import sys
import time
import threading
from typing import Callable, Optional
import numpy as np

class ProgressSpinner:
    def __init__(self, desc: str = "Processing"):
        self.desc = desc
        self.spinning = False
        self.spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self.current = 0
        self.thread: Optional[threading.Thread] = None
        
    def spin(self):
        while self.spinning:
            sys.stdout.write(f"\r{self.desc} {self.spinner_chars[self.current]} ")
            sys.stdout.flush()
            self.current = (self.current + 1) % len(self.spinner_chars)
            time.sleep(0.1)
        sys.stdout.write("\r" + " " * (len(self.desc) + 2) + "\r")
        sys.stdout.flush()
    
    def start(self):
        self.spinning = True
        self.thread = threading.Thread(target=self.spin)
        self.thread.start()
    
    def stop(self):
        self.spinning = False
        if self.thread:
            self.thread.join()

class AudioLevelMeter:
    def __init__(self, width: int = 40):
        self.width = width
        
    def draw(self, level: float):
        """Draw an audio level meter with the given level (0.0 to 1.0)"""
        level = min(1.0, max(0.0, level))
        filled = int(self.width * level)
        meter = "▰" * filled + "▱" * (self.width - filled)
        peak = "█" if level > 0.9 else "▢" if level > 0.6 else "▫"
        return f"[{meter}] {peak} {level:>4.2f}"

class RecordingProgress:
    def __init__(self, duration: float, update_interval: float = 0.1):
        self.duration = duration
        self.update_interval = update_interval
        self.start_time = None
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.level_meter = AudioLevelMeter()
        
    def update(self, audio_callback: Callable[[], np.ndarray]):
        while self.running:
            if self.start_time is None:
                self.start_time = time.time()
            
            elapsed = time.time() - self.start_time
            remaining = max(0, self.duration - elapsed)
            progress = min(1.0, elapsed / self.duration)
            
            # Get current audio level from callback
            try:
                current_audio = audio_callback()
                if len(current_audio) > 0:
                    level = float(np.max(np.abs(current_audio[-1000:])))
                else:
                    level = 0.0
            except:
                level = 0.0
            
            # Draw progress
            meter = self.level_meter.draw(level)
            sys.stdout.write(f"\rRecording: {elapsed:>4.1f}s / {self.duration:.1f}s [{int(progress*100):>3}%] {meter}")
            sys.stdout.flush()
            
            if elapsed >= self.duration:
                break
                
            time.sleep(self.update_interval)
        
        sys.stdout.write("\n")
        sys.stdout.flush()
    
    def start(self, audio_callback: Callable[[], np.ndarray]):
        self.running = True
        self.thread = threading.Thread(target=self.update, args=(audio_callback,))
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()