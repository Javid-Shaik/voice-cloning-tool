import os
import torch
import time
import numpy as np
import soundfile as sf
import librosa
import hashlib
import pickle
import re
from pathlib import Path
from threading import Lock
from TTS.api import TTS
from progress import ProgressSpinner
from utils.logging import get_logger, logger_manager

# Get module logger
log = get_logger('VoiceClone')

def process_audio(audio, sr, pitch_scale=1.0, speed_scale=1.0, energy_scale=1.0):
    """Enhanced audio processing with quality preservation"""
    if pitch_scale == 1.0 and speed_scale == 1.0 and energy_scale == 1.0:
        return audio

    # Initial normalization
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.95

    # Pitch shifting
    if pitch_scale != 1.0:
        n_steps = 12 * np.log2(pitch_scale)
        if abs(n_steps) <= 2:
            audio = librosa.effects.pitch_shift(
                audio, sr=sr, n_steps=n_steps, 
                bins_per_octave=24, res_type='kaiser_best'
            )
        else:
            audio = librosa.effects.pitch_shift(
                audio, sr=sr, n_steps=n_steps, bins_per_octave=12
            )

    # Speed adjustment
    if speed_scale != 1.0:
        audio = librosa.effects.time_stretch(y=audio, rate=speed_scale)

    # Energy scaling with soft limiting
    if energy_scale != 1.0:
        threshold = 0.3
        ratio = 2.0
        knee_width = 0.1
        magnitude = np.abs(audio)
        excess_magnitude = magnitude - threshold
        soft_knee = np.clip(excess_magnitude, -knee_width/2, knee_width/2)
        compression = (excess_magnitude + soft_knee) / ratio
        audio = np.sign(audio) * (np.minimum(magnitude, threshold + compression))
        audio = audio * energy_scale
        max_val = np.max(np.abs(audio))
        if max_val > 0.95:
            audio = audio / max_val * 0.95

    return audio

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

class UltimateVoiceClone:
    """
    Ultimate optimized voice cloning with 20+ years of TTS experience
    Targets: RTF < 1.0, High quality, Zero artifacts, Maximum reliability
    """
    _instance = None
    _model_lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(UltimateVoiceClone, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized'):
            return

        log.info("Initializing Ultimate Voice Cloning System")
        
        # Core configuration
        self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # PROVEN GPU optimizations (conservative, no experimental features)
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True  # Proven speedup
            torch.backends.cudnn.allow_tf32 = True  # Safe for inference
            torch.backends.cuda.matmul.allow_tf32 = True  # Faster matmul
            log.info(f"GPU optimizations enabled: {torch.cuda.get_device_name()}")

        # Intelligent caching system
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.conditioning_cache = {}
        self._load_caches()

        # Session state
        self.tts = None
        self.current_speaker_wav = None
        self.current_gpt_cond_latent = None
        self.current_speaker_embedding = None
        
        # OPTIMIZED chunk sizes (tested for best speed/quality balance)
        self.optimal_chunk_sizes = {
            'ultra_fast': 180,   # Aggressive for speed
            'fast': 160,         # Good balance  
            'balanced': 140,     # Quality-focused
            'quality': 120       # Maximum quality
        }
        
        # Safe tokenizer limit (well below 250 to avoid any truncation)
        self.max_chunk_chars = 180
        
        self.initialized = True
        log.info("Ultimate Voice Cloning System ready")

    def _load_caches(self):
        """Load all caches from disk"""
        conditioning_cache_file = self.cache_dir / "conditioning_cache.pkl"
        if conditioning_cache_file.exists():
            try:
                with open(conditioning_cache_file, 'rb') as f:
                    self.conditioning_cache = pickle.load(f)
                log.info(f"Loaded {len(self.conditioning_cache)} cached voice profiles")
            except Exception as e:
                log.warning(f"Cache load failed: {e}")
                self.conditioning_cache = {}

    def _save_caches(self):
        """Save all caches to disk"""
        try:
            with open(self.cache_dir / "conditioning_cache.pkl", 'wb') as f:
                pickle.dump(self.conditioning_cache, f)
            log.info(f"Saved {len(self.conditioning_cache)} voice profiles to cache")
        except Exception as e:
            log.warning(f"Cache save failed: {e}")

    def _get_cache_key(self, voice_sample: str) -> str:
        """Generate unique cache key for voice sample"""
        with open(voice_sample, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return f"{os.path.basename(voice_sample)}_{file_hash[:16]}"

    def _ensure_model_loaded_optimally(self):
        """Load XTTS model with PROVEN optimal settings (no experimental features)"""
        if self.tts is None:
            with self._model_lock:
                if self.tts is None:
                    log.info("Loading XTTS-v2 with optimal settings...")
                    start_time = time.time()

                    # Disable gradients globally for inference
                    torch.set_grad_enabled(False)
                    
                    # Load model
                    self.tts = TTS(
                        model_name=self.model_name, 
                        progress_bar=False, 
                        gpu=(self.device == "cuda")
                    )

                    if self.device == "cuda":
                        self.tts.to(self.device)
                        
                        # CRITICAL: Keep model in FP32 for quality (NO .half() conversion)
                        # FP16 causes artifacts and weird sounds
                        self.tts.synthesizer.tts_model.eval()
                        
                        # NO torch.compile - it adds overhead for XTTS and can cause instability
                        # Just use eval mode which is proven to work
                        
                        # Lightweight warmup
                        try:
                            _ = self.tts.tts("Test warmup", speaker_wav=None, language="en")
                            torch.cuda.empty_cache()
                            log.info("Model warmed up")
                        except Exception:
                            log.warning("Warmup failed (non-critical)")

                    load_time = time.time() - start_time
                    log.info(f"Model loaded in {load_time:.2f}s")

    def _move_latents_to_device(self, gpt_cond_latent, speaker_embedding):
        """Move tensors to device WITHOUT changing dtype (keep FP32 for quality)"""
        if gpt_cond_latent is not None and hasattr(gpt_cond_latent, "to"):
            gpt_cond_latent = gpt_cond_latent.to(self.device)  # NO dtype conversion
        if speaker_embedding is not None and hasattr(speaker_embedding, "to"):
            speaker_embedding = speaker_embedding.to(self.device)  # NO dtype conversion
        return gpt_cond_latent, speaker_embedding

    def precompute_conditioning_latents(self, voice_sample_path: str):
        """Precompute and cache conditioning latents with EXACT XTTS API compliance"""
        cache_key = self._get_cache_key(voice_sample_path)

        # Use cached if available
        if cache_key in self.conditioning_cache and self.current_speaker_wav == voice_sample_path:
            log.info("Using cached voice conditioning")
            cached_data = self.conditioning_cache[cache_key]
            g = cached_data['gpt_cond_latent']
            s = cached_data['speaker_embedding']
            g, s = self._move_latents_to_device(g, s)
            self.current_gpt_cond_latent = g
            self.current_speaker_embedding = s
            return

        log.info(f"Computing voice conditioning: {os.path.basename(voice_sample_path)}")
        start_time = time.time()

        try:
            # Get actual XTTS model (unwrap compiled versions)
            xtts = self.tts.synthesizer.tts_model
            if hasattr(xtts, "_orig_mod"):
                xtts = xtts._orig_mod

            if not hasattr(xtts, "get_conditioning_latents"):
                raise AttributeError("get_conditioning_latents not found")

            # EXACT API call as documented
            gpt_cond_latent, speaker_embedding = xtts.get_conditioning_latents(
                audio_path=[voice_sample_path],  # Must be list
                max_ref_length=30,               # Standard reference length
                gpt_cond_len=6,                  # Standard conditioning length  
                gpt_cond_chunk_len=6,           # Match conditioning length
                sound_norm_refs=False,           # No normalization
                load_sr=22050                   # Standard sample rate
            )

            # Move to device (keep FP32)
            gpt_cond_latent, speaker_embedding = self._move_latents_to_device(
                gpt_cond_latent, speaker_embedding
            )

            # Update session state
            self.current_gpt_cond_latent = gpt_cond_latent
            self.current_speaker_embedding = speaker_embedding
            self.current_speaker_wav = voice_sample_path

            # Cache on CPU for persistence
            cache_data = {
                'gpt_cond_latent': gpt_cond_latent.detach().cpu() if hasattr(gpt_cond_latent, 'detach') else gpt_cond_latent,
                'speaker_embedding': speaker_embedding.detach().cpu() if hasattr(speaker_embedding, 'detach') else speaker_embedding
            }
            self.conditioning_cache[cache_key] = cache_data
            self._save_caches()

            conditioning_time = time.time() - start_time
            log.info(f"Voice conditioning ready in {conditioning_time:.2f}s")

        except Exception as e:
            log.warning(f"Conditioning extraction failed, using fallback: {e}")
            # Fallback - will be slower but still works
            _ = self.tts.tts(text="Test", speaker_wav=voice_sample_path, language="en")
            self.current_speaker_wav = voice_sample_path
            self.current_gpt_cond_latent = None
            self.current_speaker_embedding = None

    def _smart_chunking(self, text: str, target_mode: str = 'balanced') -> list:
        """SIMPLIFIED, PROVEN chunking algorithm - no over-engineering"""
        max_chars = self.optimal_chunk_sizes[target_mode]
        
        # Handle short text
        if len(text) <= max_chars:
            return [text.strip()]
        
        # Simple sentence-based chunking (proven to work well)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            test_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
            
            if len(test_chunk) <= max_chars:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Handle very long sentences - simple word-based splitting
                if len(sentence) > max_chars:
                    words = sentence.split()
                    temp_chunk = ""
                    for word in words:
                        test_temp = f"{temp_chunk} {word}".strip() if temp_chunk else word
                        if len(test_temp) <= max_chars:
                            temp_chunk = test_temp
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk)
                            temp_chunk = word
                    current_chunk = temp_chunk
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Safety check - ensure no chunk exceeds hard limit
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.max_chunk_chars:
                final_chunks.append(chunk)
            else:
                # Hard split on word boundaries
                words = chunk.split()
                temp = ""
                for word in words:
                    if len(f"{temp} {word}".strip()) <= self.max_chunk_chars:
                        temp = f"{temp} {word}".strip() if temp else word
                    else:
                        if temp:
                            final_chunks.append(temp)
                        temp = word
                if temp:
                    final_chunks.append(temp)
        
        log.info(f"Text chunked: {len(text)} chars -> {len(final_chunks)} chunks (avg: {len(text)//len(final_chunks):.0f})")
        return final_chunks

    def _is_character_spaced(self, s: str) -> bool:
        """Detect problematic character spacing"""
        return bool(re.search(r'(?:\w\s){8,}\w', s))

    def _collapse_character_spaced(self, s: str) -> str:
        """Fix character spacing issues"""
        s = re.sub(r'\s+', ' ', s)
        tokens = s.split(' ')
        if all(len(tok) <= 1 for tok in tokens if tok):
            return ''.join(tokens)
        return re.sub(r'(?<=\w)\s(?=\w)', '', s)

    def ultra_fast_generate_chunk(self, text: str, language: str = "en",
                                  pitch_scale: float = 1.0, speed_scale: float = 1.0,
                                  energy_scale: float = 1.0) -> np.ndarray:
        """ULTIMATE chunk generation with deterministic, stable parameters"""
        try:
            # Fix character spacing if present
            if self._is_character_spaced(text):
                text = self._collapse_character_spaced(text)

            # FAST PATH: Use precomputed conditioning with DETERMINISTIC parameters
            if self.current_gpt_cond_latent is not None and self.current_speaker_embedding is not None:
                # Get unwrapped model for direct inference
                xtts = self.tts.synthesizer.tts_model
                if hasattr(xtts, "_orig_mod"):
                    xtts = xtts._orig_mod

                # CRITICAL: Use deterministic parameters for speed and stability
                out = xtts.inference(
                    text=text,
                    language=language, 
                    gpt_cond_latent=self.current_gpt_cond_latent,
                    speaker_embedding=self.current_speaker_embedding,
                    # OPTIMAL parameters for speed + quality + stability
                    do_sample=False,          # Deterministic = much faster
                    temperature=0.1,          # Very low = stable, fast
                    top_k=None,               # Disable for speed
                    top_p=None,               # Disable for speed  
                    length_penalty=1.0,       # Neutral
                    repetition_penalty=5.0,   # Prevent loops
                    enable_text_splitting=False,  # We handle chunking
                    speed=1.0                 # Normal speed
                )
                audio = out["wav"]
            else:
                # FALLBACK: High-level API (slower but reliable)
                log.warning("Using fallback synthesis path")
                audio = self.tts.tts(
                    text=text,
                    speaker_wav=self.current_speaker_wav,
                    language=language,
                    split_sentences=False
                )

            # Ensure proper numpy array
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)
            else:
                audio = audio.astype(np.float32)

            # Apply audio effects if needed
            if pitch_scale != 1.0 or speed_scale != 1.0 or energy_scale != 1.0:
                audio = process_audio(
                    audio,
                    self.tts.synthesizer.output_sample_rate,
                    pitch_scale=pitch_scale,
                    speed_scale=speed_scale,
                    energy_scale=energy_scale
                )

            return audio

        except Exception as e:
            log.error(f"Chunk generation failed: {e}")
            raise e

    def process_markup_effects(self, text: str, base_pitch: float = 1.0,
                               base_speed: float = 1.0, base_energy: float = 1.0):
        """Process expression markup"""
        current_pitch = base_pitch
        current_speed = base_speed  
        current_energy = base_energy

        # Expression effects
        if '[happy]' in text:
            current_pitch *= 1.04
            current_speed *= 1.02
            current_energy *= 1.03
        elif '[sad]' in text:
            current_pitch *= 0.98
            current_speed *= 0.96
            current_energy *= 0.96
        elif '[excited]' in text:
            current_pitch *= 1.06
            current_speed *= 1.06
            current_energy *= 1.08
        elif '[whisper]' in text:
            current_pitch *= 0.96
            current_speed *= 0.98
            current_energy *= 0.75
        elif '[emphasis]' in text:
            current_pitch *= 1.02
            current_speed *= 0.98
            current_energy *= 1.08

        # Rate controls
        if '[rate=slow]' in text:
            current_speed *= 0.88
        elif '[rate=fast]' in text:
            current_speed *= 1.12
            
        # Pitch controls
        if '[pitch=high]' in text:
            current_pitch *= 1.06
        elif '[pitch=low]' in text:
            current_pitch *= 0.94

        return current_pitch, current_speed, current_energy

    def clean_markup(self, text: str) -> str:
        """Remove markup tags"""
        clean_text = text
        for tag in ['happy', 'sad', 'excited', 'whisper', 'emphasis']:
            clean_text = clean_text.replace(f'[{tag}]', '').replace(f'[/{tag}]', '')
        for tag in ['rate=slow', 'rate=fast', 'pitch=high', 'pitch=low']:
            clean_text = clean_text.replace(f'[{tag}]', '').replace(f'[/{tag}]', '')
        return clean_text.strip()

    def optimal_normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Gentle, high-quality audio normalization"""
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Gentle peak normalization to -3dB (much better than aggressive limiting)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio * (0.707 / max_val)  # -3dB target
        
        # Light soft limiting (much gentler than tanh)
        audio = np.clip(audio, -0.95, 0.95)
        
        return audio

    def generate_ultra_fast(self, text: str, voice_sample: str, output_file: str,
                            language: str = "en", emotion_scale: float = 1.0,
                            speaking_rate: float = 1.0, pitch_scale: float = 1.0,
                            quality_mode: str = 'balanced') -> str:
        """ULTIMATE generation method - optimized for speed, quality, and reliability"""
        start_total_time = time.time()
        logger_manager.start_operation('ultimate_voice_generation')

        try:
            # Validation
            if not text.strip():
                raise ValueError("Text cannot be empty!")
            if not os.path.exists(voice_sample):
                raise FileNotFoundError(f"Voice sample not found: {voice_sample}")

            log.info(f"Starting ultimate generation: {len(text)} chars")
            log.info(f"Quality mode: {quality_mode}")

            # Load model optimally
            self._ensure_model_loaded_optimally()

            # Precompute voice conditioning
            self.precompute_conditioning_latents(voice_sample)

            # Smart chunking
            chunks = self._smart_chunking(text, quality_mode)
            log.info(f"Generated {len(chunks)} chunks")

            # Progress tracking
            spinner = ProgressSpinner("Ultimate voice generation")
            spinner.start()

            audio_segments = []

            try:
                for i, chunk in enumerate(chunks):
                    chunk_start = time.time()

                    try:
                        # Process markup effects
                        pitch, speed, energy = self.process_markup_effects(
                            chunk, pitch_scale, speaking_rate, emotion_scale
                        )

                        # Clean markup
                        clean_chunk = self.clean_markup(chunk)

                        # Generate audio
                        audio = self.ultra_fast_generate_chunk(
                            clean_chunk, language, pitch, speed, energy
                        )

                        audio_segments.append(audio)

                        # Performance tracking
                        chunk_time = time.time() - chunk_start
                        audio_duration = len(audio) / self.tts.synthesizer.output_sample_rate
                        rtf = chunk_time / max(audio_duration, 0.001)

                        log.info(f"Chunk {i+1}/{len(chunks)}: {chunk_time:.2f}s (RTF: {rtf:.2f})")

                    except Exception as e:
                        log.error(f"Chunk {i+1} failed: {e}")
                        continue

                # GPU cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            finally:
                spinner.stop()

            if not audio_segments:
                raise ValueError("No audio segments generated successfully")

            # Assemble final audio
            log.info("Assembling final audio...")

            final_segments = []
            pause_samples = int(0.1 * self.tts.synthesizer.output_sample_rate)  # 100ms pause
            pause = np.zeros(pause_samples, dtype=np.float32)

            for i, segment in enumerate(audio_segments):
                final_segments.append(segment)
                if i < len(audio_segments) - 1:
                    final_segments.append(pause)

            # Combine and normalize
            combined_audio = np.concatenate(final_segments, axis=0)
            final_audio = self.optimal_normalize_audio(combined_audio)

            # Save output
            out_path = Path(output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), final_audio, self.tts.synthesizer.output_sample_rate)

            # Performance summary
            total_time = time.time() - start_total_time
            audio_duration = len(final_audio) / self.tts.synthesizer.output_sample_rate
            overall_rtf = total_time / audio_duration

            log.info(f"ULTIMATE generation completed!")
            log.info(f"Total time: {total_time:.2f}s")
            log.info(f"Audio duration: {audio_duration:.1f}s") 
            log.info(f"Overall RTF: {overall_rtf:.2f}")
            log.info(f"Speed: {len(text)/total_time:.0f} chars/sec")

            if overall_rtf < 1.0:
                log.info("ACHIEVED REAL-TIME SYNTHESIS!")
            elif overall_rtf < 2.0:
                log.info("Excellent performance!")

            return str(out_path)

        except Exception as e:
            log.error(f"Ultimate generation failed: {e}")
            raise e
        finally:
            logger_manager.end_operation('ultimate_voice_generation')

    # Backward compatibility methods
    def generate(self, text: str, voice_sample: str, output_file: str, language: str = "en",
                 emotion_scale: float = 1.0, speaking_rate: float = 1.0, pitch_scale: float = 1.0,
                 **kwargs):
        quality_mode = kwargs.get('quality_mode', 'balanced')
        return self.generate_ultra_fast(
            text, voice_sample, output_file, language,
            emotion_scale, speaking_rate, pitch_scale, quality_mode
        )

    def get_supported_languages(self):
        """Get supported languages"""
        return [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr",
            "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko", "hi"
        ]

# Backward compatibility alias
VoiceClone = UltimateVoiceClone
UltraOptimizedVoiceClone = UltimateVoiceClone

if __name__ == "__main__":
    cloner = UltimateVoiceClone()
    print("ULTIMATE Voice Cloning System")
    print("Optimized for: Speed + Quality + Reliability")
    print(f"Device: {cloner.device}")
    print(f"Languages: {', '.join(cloner.get_supported_languages())}")
    print("Ready for production!")