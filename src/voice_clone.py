"""
Voice Cloner Class for Coqui TTS 0.27.1
Compatible with XTTS-v2 model
"""

import os
import torch
import time
import numpy as np
import soundfile as sf
import librosa
import hashlib
import json
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from TTS.api import TTS
from progress import ProgressSpinner
from utils.logging import get_logger, logger_manager

# Get module logger
log = get_logger('VoiceClone')

def process_audio(audio, sr, pitch_scale=1.0, speed_scale=1.0, energy_scale=1.0):
    """
    Enhanced audio processing with quality preservation
    
    Args:
        audio (np.array): Input audio signal
        sr (int): Sample rate
        pitch_scale (float): Pitch adjustment factor (1.0 = no change)
        speed_scale (float): Speed adjustment factor (1.0 = no change)
        energy_scale (float): Volume adjustment factor (1.0 = no change)
        
    Returns:
        np.array: Processed audio signal
    """
    # Skip processing if no changes needed
    if pitch_scale == 1.0 and speed_scale == 1.0 and energy_scale == 1.0:
        return audio
        
    # Initial normalization
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95
    
    # Higher quality pitch shifting with better formant preservation
    if pitch_scale != 1.0:
        # Convert scale to semitones
        n_steps = 12 * np.log2(pitch_scale)
        
        # Use higher quality settings for small pitch changes
        if abs(n_steps) <= 2:  # For subtle changes
            audio = librosa.effects.pitch_shift(
                audio,
                sr=sr,
                n_steps=n_steps,
                bins_per_octave=24,  # More bins for smoother shifts
                res_type='kaiser_best'  # Higher quality resampling
            )
        else:  # For larger changes
            audio = librosa.effects.pitch_shift(
                audio,
                sr=sr,
                n_steps=n_steps,
                bins_per_octave=12
            )
    
    # Improved speed adjustment
    if speed_scale != 1.0:
        # Use librosa's time stretch function
        audio = librosa.effects.time_stretch(
            y=audio,
            rate=speed_scale
        )
    
    # Enhanced volume adjustment with soft limiting
    if energy_scale != 1.0:
        # Apply a soft knee compressor for smoother dynamics
        threshold = 0.3
        ratio = 2.0
        knee_width = 0.1
        
        # Compress before scaling
        magnitude = np.abs(audio)
        excess_magnitude = magnitude - threshold
        soft_knee = np.clip(excess_magnitude, -knee_width/2, knee_width/2)
        compression = (excess_magnitude + soft_knee) / ratio
        audio = np.sign(audio) * (np.minimum(magnitude, threshold + compression))
        
        # Apply energy scaling
        audio = audio * energy_scale
        
        # Final normalization with headroom
        max_val = np.max(np.abs(audio))
        if max_val > 0.95:
            audio = audio / max_val * 0.95
        
    return audio

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

class VoiceClone:
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to ensure only one model instance"""
        if cls._instance is None:
            cls._instance = super(VoiceClone, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        Initialize TTS model with XTTS-v2 for proper voice cloning support.
        Uses lazy loading and caching for better performance.
        Compatible with coqui-tts==0.27.1
        """
        if hasattr(self, 'initialized'):
            return
            
        log.info("Initializing Voice Cloning with Coqui TTS 0.27.1")
        
        # Use XTTS-v2 model which supports voice cloning
        self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        
        # Get device and optimize for inference
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            # Set optimal CUDA settings
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.9)  # Use 90% of GPU memory
            torch.cuda.set_device(0)  # Ensure using primary GPU
        print(f"Using device: {self.device}")
        
        # Set up advanced caching
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Separate caches for different types of data
        self.embedding_cache = {}
        self.audio_cache = {}  # Cache for processed audio segments
        self.model_cache = {}  # Cache for model intermediate outputs
        self._load_embedding_cache()
        
        # Optimize parallel processing
        cpu_count = os.cpu_count() or 2
        self.max_workers = min(cpu_count * 2, 8)  # 2 threads per CPU core, max 8
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="tts_worker"
        )
        
        # Initialize memory pools for better memory management
        self.inference_pool = []  # Pool of pre-allocated tensors
        
        # Don't load model yet - lazy load on first use
        self.tts = None
        self.initialized = True
        print("Voice cloning initialized with advanced optimizations")
        
    def _get_cache_key(self, voice_sample: str) -> str:
        """Generate a unique cache key for a voice sample"""
        with open(voice_sample, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return f"{os.path.basename(voice_sample)}_{file_hash}"
        
    def _load_embedding_cache(self):
        """Load cached embeddings from disk"""
        cache_file = self.cache_dir / "embedding_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                print(f"Loaded {len(self.embedding_cache)} cached voice embeddings")
            except Exception as e:
                print(f"Warning: Could not load embedding cache: {e}")
                self.embedding_cache = {}
                
    def _save_embedding_cache(self):
        """Save embeddings cache to disk"""
        cache_file = self.cache_dir / "embedding_cache.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
        except Exception as e:
            print(f"Warning: Could not save embedding cache: {e}")
        
    def _ensure_model_loaded(self):
        """Lazy load the model only when needed with optimizations"""
        if self.tts is None:
            print("Loading XTTS-v2 model...")
            
            # Enable faster inference optimizations
            torch.set_grad_enabled(False)
            if self.device == "cuda":
                torch.backends.cudnn.benchmark = True
                
            # Load model with INT8 quantization for faster inference
            self.tts = TTS(model_name=self.model_name)
            
            # Move to device and optimize
            self.tts.to(self.device)
            if self.device == "cuda":
                self.tts.synthesizer.tts_model.half()  # Use FP16 for faster GPU inference
                
            print("XTTS-v2 model loaded successfully with optimizations enabled")
            
    def _chunk_text(self, text: str, max_tokens: int = 240):
        """
        Split text into safe chunks under XTTS character limit.
        """
        import re
        sentences = re.split(r'(?<=[.!?]) +', text)  # Split on sentence boundaries
        chunks, current = [], ""
        
        for s in sentences:
            if len(current) + len(s) + 1 <= max_tokens:
                current += " " + s if current else s
            else:
                if current:
                    chunks.append(current.strip())
                if len(s) > max_tokens:
                    # Hard split very long sentences
                    for i in range(0, len(s), max_tokens):
                        chunks.append(s[i:i+max_tokens].strip())
                    current = ""
                else:
                    current = s
        if current:
            chunks.append(current.strip())
        
        return chunks


    def generate(self, text: str, voice_sample: str, output_file: str, language: str = "en", 
                emotion_scale: float = 1.0, speaking_rate: float = 1.0, pitch_scale: float = 1.0,
                speaker_consistency: float = 0.75, batch_size: int = 3):
        """
        Generate expressive speech in cloned voice using XTTS-v2.
        
        The model supports expression control through text markup:
        [happy]text[/happy], [sad]text[/sad], [excited]text[/excited]
        [whisper]text[/whisper], [emphasis]text[/emphasis]
        
        speaker_consistency: Controls how closely the output matches the reference voice (0.5-1.0)
            Higher values (>0.7) maintain closer similarity to the reference voice
            Lower values allow more expressive variation but may reduce voice similarity
        batch_size: Number of sentences to process in parallel (default: 3)
            Higher values may be faster but use more memory
        
        Args:
            text (str): Text to synthesize (can include expression markup)
            voice_sample (str): Path to reference audio file
            output_file (str): Path for output audio file
            language (str): Language code (en, es, fr, etc.)
            emotion_scale (float): Scale factor for emotional expression (0.5-1.5)
            speaking_rate (float): Speech rate multiplier (0.5-2.0)
            pitch_scale (float): Pitch adjustment scale (0.5-2.0)
        """
        # Start overall operation timing
        logger_manager.start_operation('voice_generation')
        try:
            # Validate inputs
            if not text.strip():
                log.error("Empty text provided")
                raise ValueError("Text to synthesize is empty!")
                
            if not os.path.exists(voice_sample):
                log.error(f"Voice sample not found: {voice_sample}")
                raise FileNotFoundError(f"Voice sample not found: {voice_sample}")
                
            log.info(f"Cloning voice from: {voice_sample}")
            log.info(f"Text length: {len(text)} chars, Language: {language}")
            log.debug(f"Text preview: {text[:100]}{'...' if len(text) > 100 else ''}")
            log.debug(f"Output file: {output_file}")
            
            # Log generation parameters
            log.info(f"Generation parameters - Emotion: {emotion_scale:.2f}, "
                    f"Rate: {speaking_rate:.2f}, Pitch: {pitch_scale:.2f}, "
                    f"Consistency: {speaker_consistency:.2f}")
        except (ValueError, FileNotFoundError) as e:
            log.error(f"Input validation failed: {str(e)}")
            raise

        try:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Show progress spinner during generation
            spinner = ProgressSpinner("Generating speech with cloned voice")
            spinner.start()
            
            try:
                # Ensure model is loaded
                self._ensure_model_loaded()
                
                # Use cached speaker embedding or compute new one
                cache_key = self._get_cache_key(voice_sample)
                if cache_key in self.embedding_cache:
                    print("\nUsing cached voice embedding...")
                    speaker_embedding = self.embedding_cache[cache_key]
                else:
                    print("\nComputing and caching voice embedding...")
                    try:
                        # Generate speaker embedding from reference audio
                        speaker_embedding = self.tts.synthesizer.tts_model.speaker_manager.compute_embedding(voice_sample)
                        # Cache the embedding for future use
                        self.embedding_cache[cache_key] = speaker_embedding
                        self._save_embedding_cache()
                    except Exception as e:
                        print(f"Warning: Could not compute speaker embedding: {e}")
                        speaker_embedding = None
                
                # Simple and fast sentence splitting
                print("\nPreparing text...")
                # Split on common sentence endings while preserving punctuation
                print("\nPreparing text...")
                sentences = self._chunk_text(text, max_tokens=248)
                total_sentences = len(sentences)
                print(f"Processing {total_sentences} chunks...")

                temp_outputs = []
                
                # Process sentences sequentially for better stability
                for sentence_idx, sentence in enumerate(sentences):
                        if not sentence:
                            continue
                        
                        sentence = sentence.strip()
                        if sentence:
                            # Use sentence index for temp file naming
                            current_pitch = pitch_scale
                            current_speed = speaking_rate
                            current_energy = 1.0 * emotion_scale
                        
                        # Handle expression markup with minimal, natural adjustments
                        if '[happy]' in sentence:
                            current_pitch *= 1.05  # Very slight pitch increase
                            current_speed *= 1.03  # Barely faster
                            current_energy *= 1.05 # Slight energy boost
                        elif '[sad]' in sentence:
                            current_pitch *= 0.98  # Very slight pitch decrease
                            current_speed *= 0.95  # Slightly slower
                            current_energy *= 0.95 # Slightly softer
                        elif '[excited]' in sentence:
                            current_pitch *= 1.08  # Moderate pitch increase
                            current_speed *= 1.08  # Moderately faster
                            current_energy *= 1.1  # Moderate energy boost
                        elif '[whisper]' in sentence:
                            current_pitch *= 0.95  # Slight pitch decrease
                            current_speed *= 0.97  # Very slightly slower
                            current_energy *= 0.7  # Softer but not too quiet
                        elif '[emphasis]' in sentence:
                            current_pitch *= 1.02  # Minimal pitch change
                            current_speed *= 0.98  # Very slightly slower
                            current_energy *= 1.1  # Moderate emphasis
                        
                        # Handle rate/pitch controls with gentler adjustments
                        if '[rate=slow]' in sentence:
                            current_speed *= 0.85  # Less extreme slowdown
                        elif '[rate=fast]' in sentence:
                            current_speed *= 1.15  # Less extreme speedup
                        if '[pitch=high]' in sentence:
                            current_pitch *= 1.08  # Subtle pitch increase
                        elif '[pitch=low]' in sentence:
                            current_pitch *= 0.92  # Subtle pitch decrease
                        
                        # Remove markup for TTS
                        clean_sentence = sentence
                        for tag in ['happy', 'sad', 'excited', 'whisper', 'emphasis']:
                            clean_sentence = clean_sentence.replace(f'[{tag}]', '').replace(f'[/{tag}]', '')
                        for tag in ['rate=slow', 'rate=fast', 'pitch=high', 'pitch=low']:
                            clean_sentence = clean_sentence.replace(f'[{tag}]', '').replace(f'[/{tag}]', '')
                        
                        temp_file = f"{output_file}.temp{sentence_idx}.wav"
                        print(f"Generating sentence {sentence_idx + 1}/{total_sentences}")
                        
                        # Generate and process audio in memory
                        audio = self.tts.tts(
                            text=clean_sentence,
                            speaker_wav=voice_sample,
                            language=language,
                            split_sentences=False,  # Important: prevent double splitting
                        )
                        
                        # Apply effects in memory without saving/loading
                        if current_pitch != 1.0 or current_speed != 1.0 or current_energy != 1.0:
                            audio = process_audio(
                                audio,
                                self.tts.synthesizer.output_sample_rate,
                                pitch_scale=current_pitch,
                                speed_scale=current_speed,
                                energy_scale=current_energy
                            )
                        
                        # Save processed audio
                        sf.write(temp_file, audio, self.tts.synthesizer.output_sample_rate)
                        
                        temp_outputs.append(temp_file)
                        print("✓", end=' ', flush=True)
                
                # Combine all sentences with proper spacing
                if len(temp_outputs) > 1:
                    print("\nMerging sentences...")
                    from pydub import AudioSegment
                    
                    combined = AudioSegment.from_wav(temp_outputs[0])
                    for temp in temp_outputs[1:]:
                        # Add slight pause between sentences
                        combined = combined + AudioSegment.silent(duration=200)
                        combined = combined + AudioSegment.from_wav(temp)
                    
                    combined.export(output_file, format="wav")
                    
                    # Clean up temp files
                    for temp in temp_outputs:
                        try:
                            os.remove(temp)
                        except:
                            pass
                else:
                    # Single sentence, just rename
                    import shutil
                    shutil.move(temp_outputs[0], output_file)
                
            finally:
                spinner.stop()
            
            log.info("Voice cloning completed successfully!")
            
            # Validate output and compute quality metrics
            try:
                logger_manager.start_operation('quality_analysis')
                audio, sr = sf.read(output_file)
                duration = len(audio) / sr
                rms_level = float(np.sqrt(np.mean(audio ** 2)))
                
                # Compute additional metrics
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
                
                # Log quality metrics
                log.info("Output Quality Metrics:", extra={
                    'metrics': {
                        'duration': f"{duration:.1f}s",
                        'sample_rate': f"{sr}Hz",
                        'rms_level': f"{rms_level:.3f}",
                        'spectral_centroid': f"{spectral_centroid:.1f}Hz",
                        'tempo': f"{tempo:.1f}BPM"
                    }
                })
                
                # Performance metrics for the entire operation
                logger_manager.end_operation('voice_generation')
                perf_metrics = logger_manager.get_performance_metrics('voice_generation')
                log.info(f"Total generation time: {perf_metrics.get('average', 0):.2f} seconds")
                
                # Warn if audio levels are too high/low
                if rms_level > 0.8:
                    log.warning("Audio levels are high and may clip")
                elif rms_level < 0.1:
                    log.warning("Audio levels are low and may be hard to hear")
                    
            except Exception as e:
                log.error("Failed to analyze output audio", exc_info=True)
                log.info("The file was generated but quality metrics are unavailable")
            finally:
                logger_manager.end_operation('quality_analysis')
                metrics = logger_manager.get_performance_metrics('quality_analysis')
                log.debug(f"Quality analysis time: {metrics.get('average', 0):.2f} seconds")
            
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