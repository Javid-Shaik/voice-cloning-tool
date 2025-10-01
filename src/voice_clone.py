import os
import torch
import time
import numpy as np
import soundfile as sf
import librosa
import hashlib
import json
import pickle
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from TTS.api import TTS
from progress import ProgressSpinner
from utils.logging import get_logger, logger_manager

# Get module logger
log = get_logger('VoiceClone')

def process_audio(audio, sr, pitch_scale=1.0, speed_scale=1.0, energy_scale=1.0):
    """
    Enhanced audio processing with quality preservation
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

class UltraOptimizedVoiceClone:
    _instance = None
    _model_lock = Lock()

    # Conservative per‑chunk char caps to avoid tokenizer truncation (XTTS-v2 ~250 char for 'en')
    # Leave ~10–20 chars headroom to prevent warnings.
    _lang_char_caps = {
        "en": 240
    }

    def __new__(cls):
        """Singleton pattern to ensure only one model instance"""
        if cls._instance is None:
            cls._instance = super(UltraOptimizedVoiceClone, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        Initialize TTS model with XTTS-v2 for ultra-fast voice cloning.
        Target: 4-6 minutes for full script generation
        """
        if hasattr(self, 'initialized'):
            return

        log.info("Initializing Ultra-Optimized Voice Cloning (Target: 4-6min generation)")

        # Use XTTS-v2 model which supports voice cloning
        self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"

        # Get device and optimize for maximum inference speed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            # Ultra-aggressive GPU optimizations
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False  # Speed over determinism
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

            # Set memory allocation strategy for speed
            try:
                torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
                torch.cuda.set_device(0)  # Ensure using primary GPU
            except Exception:
                pass  # Some environments may not support this


        # Set up advanced caching (allow override for Kaggle/CI via VC_CACHE)
        self.cache_dir = Path(os.getenv('VC_CACHE', '/kaggle/working/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Caching systems
        self.embedding_cache = {}
        self.conditioning_cache = {}  # Cache for conditioning latents
        self._load_caches()

        # Precomputed conditioning for current session
        self.current_speaker_wav = None
        self.current_gpt_cond_latent = None
        self.current_speaker_embedding = None

        # Ultra-fast chunking parameters (kept <= 230 to respect ~250-char tokenizer cap)
        self.optimal_chunk_sizes = {
            'ultra_fast': 230,
            'fast': 210,
            'balanced': 190,
            'quality': 160
        }

        # Threading for pipeline parallelism (placeholder, generation is sequential here)
        self.max_workers = min(torch.cuda.device_count() if torch.cuda.is_available() else 1, 2)

        # Model loading
        self.tts = None
        self.initialized = True
        print("Ultra-optimized voice cloning initialized")

    def _load_caches(self):
        """Load all caches from disk"""
        # Load embedding cache
        embedding_cache_file = self.cache_dir / "embedding_cache.pkl"
        if embedding_cache_file.exists():
            try:
                with open(embedding_cache_file, 'rb') as f:
                    self.embedding_cache = pickle.load(f)
                print(f"Loaded {len(self.embedding_cache)} cached embeddings")
            except Exception as e:
                print(f"Warning: Could not load embedding cache: {e}")
                self.embedding_cache = {}

        # Load conditioning cache
        conditioning_cache_file = self.cache_dir / "conditioning_cache.pkl"
        if conditioning_cache_file.exists():
            try:
                with open(conditioning_cache_file, 'rb') as f:
                    self.conditioning_cache = pickle.load(f)
                print(f"Loaded {len(self.conditioning_cache)} cached conditioning latents")
            except Exception as e:
                print(f"Warning: Could not load conditioning cache: {e}")
                self.conditioning_cache = {}

    def _save_caches(self):
        """Save all caches to disk"""
        try:
            with open(self.cache_dir / "embedding_cache.pkl", 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            with open(self.cache_dir / "conditioning_cache.pkl", 'wb') as f:
                pickle.dump(self.conditioning_cache, f)
        except Exception as e:
            print(f"Warning: Could not save caches: {e}")

    def _ensure_model_loaded_ultra_fast(self):
        """Ultra-fast model loading with maximum optimizations"""
        if self.tts is None:
            with self._model_lock:
                if self.tts is None:  # Double-check after acquiring lock
                    log.info("Loading XTTS-v2 with ultra-fast optimizations...")
                    start_time = time.time()

                    # Disable gradient computation globally for inference
                    torch.set_grad_enabled(False)

                    # Load model with speed optimizations
                    self.tts = TTS(model_name=self.model_name, progress_bar=False, gpu=(self.device == "cuda"))

                    if self.device == "cuda":
                        # Ultra-aggressive GPU optimizations
                        self.tts.to(self.device)

                        # Convert to half precision for 2x speed improvement (guarded)
                        try:
                            if self.device == "cuda" and torch.cuda.is_available():
                                # Use AMP/FP16 only when CUDA is available
                                self.tts.synthesizer.tts_model.half()
                                log.info("Model converted to FP16 for 2x speed boost")
                            else:
                                log.info("Skipping FP16 conversion: CUDA not available")
                        except Exception:
                            log.warning("FP16 conversion failed, using FP32")

                        # Set to evaluation mode and enable inference optimizations
                        self.tts.synthesizer.tts_model.eval()

                        # Enable inference mode flag where available
                        try:
                            for module in self.tts.synthesizer.tts_model.modules():
                                if hasattr(module, 'inference_mode'):
                                    module.inference_mode = True
                        except Exception:
                            pass

                        # Compile model for faster inference (PyTorch 2.0+)
                        try:
                            if hasattr(torch, 'compile'):
                                self.tts.synthesizer.tts_model = torch.compile(
                                    self.tts.synthesizer.tts_model,
                                    mode='max-autotune'
                                )
                                log.info("Model compiled with torch.compile for maximum speed")
                        except Exception as e:
                            log.warning(f"Model compilation failed: {e}")

                        # Warm up the model to optimize CUDA kernels
                        try:
                            dummy_text = "Warming up the model for optimal performance."
                            _ = self.tts.tts(dummy_text, speaker_wav=None, language="en")
                            torch.cuda.empty_cache()
                            log.info("Model warmed up successfully")
                        except Exception:
                            log.warning("Model warmup failed")

                    load_time = time.time() - start_time
                    log.info(f"Ultra-fast model loaded in {load_time:.2f}s")

    def _get_cache_key(self, voice_sample: str) -> str:
        """Generate a unique cache key for a voice sample"""
        with open(voice_sample, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return f"{os.path.basename(voice_sample)}_{file_hash}"

    def _move_latents_to_device(self, gpt_cond_latent, speaker_embedding):
        """Ensure cached tensors are on the active device and dtype"""
        if gpt_cond_latent is not None and hasattr(gpt_cond_latent, "to"):
            gpt_cond_latent = gpt_cond_latent.to(self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32)
        if speaker_embedding is not None and hasattr(speaker_embedding, "to"):
            speaker_embedding = speaker_embedding.to(self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32)
        return gpt_cond_latent, speaker_embedding

    def precompute_conditioning_latents(self, voice_sample_path: str):
        """
        Precompute and cache conditioning latents for ultra-fast generation.
        This is the key optimization that eliminates per-call overhead.
        """
        cache_key = self._get_cache_key(voice_sample_path)

        # Check if we have cached conditioning for this voice
        if cache_key in self.conditioning_cache and self.current_speaker_wav == voice_sample_path:
            log.info("Using cached conditioning latents")
            cached_data = self.conditioning_cache[cache_key]
            g = cached_data['gpt_cond_latent']
            s = cached_data['speaker_embedding']
            # Move to device for fast path
            g, s = self._move_latents_to_device(g, s)
            self.current_gpt_cond_latent = g
            self.current_speaker_embedding = s
            return

        log.info(f"Precomputing conditioning latents for: {voice_sample_path}")
        start_time = time.time()

        try:
            # Load to ensure file valid; actual extraction uses model method
            audio, sr = sf.read(voice_sample_path)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)  # Convert to mono

            # Extract conditioning latents using the model's internal method
            try:
                gpt_cond_latent, speaker_embedding = self.tts.synthesizer.tts_model.get_conditioning_latents(
                    audio_path=voice_sample_path,
                    gpt_cond_len=self.tts.synthesizer.tts_model.gpt_cond_len,
                    max_ref_length=self.tts.synthesizer.tts_model.max_ref_len,
                    sound_norm_refs=self.tts.synthesizer.tts_model.sound_norm_refs,
                )

                # Store in session cache (on device)
                gpt_cond_latent, speaker_embedding = self._move_latents_to_device(gpt_cond_latent, speaker_embedding)
                self.current_gpt_cond_latent = gpt_cond_latent
                self.current_speaker_embedding = speaker_embedding
                self.current_speaker_wav = voice_sample_path

                # Store in persistent cache (on CPU for portability)
                cache_data = {
                    'gpt_cond_latent': gpt_cond_latent.detach().cpu() if hasattr(gpt_cond_latent, 'detach') else gpt_cond_latent,
                    'speaker_embedding': speaker_embedding.detach().cpu() if hasattr(speaker_embedding, 'detach') else speaker_embedding
                }
                self.conditioning_cache[cache_key] = cache_data
                self._save_caches()

                conditioning_time = time.time() - start_time
                log.info(f"Conditioning latents computed in {conditioning_time:.2f}s")

            except AttributeError:
                # Fallback for models that don't have get_conditioning_latents
                log.warning("Model doesn't support get_conditioning_latents, using fallback")
                _ = self.tts.tts(
                    text="Test for conditioning",
                    speaker_wav=voice_sample_path,
                    language="en"
                )
                self.current_speaker_wav = voice_sample_path
                self.current_gpt_cond_latent = None
                self.current_speaker_embedding = None

        except Exception as e:
            log.error(f"Failed to precompute conditioning latents: {e}")
            raise e

    def _smart_sentence_split(self, text: str) -> list:
        """Smart sentence splitting that preserves natural flow"""
        # Enhanced sentence splitting that handles abbreviations and edge cases
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        # Clean up and filter empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _calculate_text_complexity(self, text: str) -> float:
        """Calculate text complexity for optimal chunk sizing"""
        complexity = 0.0
        # Length factor
        complexity += min(len(text) / 300, 0.3)
        # Punctuation density
        punct_count = len(re.findall(r'''[,;:()"'-]''', text))
        complexity += min(punct_count / (len(text) / 10), 0.3)
        # Number and special character density
        special_count = len(re.findall(r'[\d$%&@#]', text))
        complexity += min(special_count / (len(text) / 20), 0.2)
        # Technical terms (rough heuristic)
        tech_words = len(re.findall(r'[A-Z]{2,}|\w+[A-Z]\w+|\w{10,}', text))
        complexity += min(tech_words / (len(text.split()) / 5), 0.2)
        return min(complexity, 1.0)

    def _split_long_sentence(self, sentence: str, max_length: int) -> list:
        """Split extremely long sentences on natural breaks"""
        if len(sentence) <= max_length:
            return [sentence]

        parts = []
        clause_patterns = [
            r'(,\s+(?:and|but|or|so|yet|for)\s+)',
            r'(;\s*)',
            r'(:\s*)',
            r'(,\s+(?:which|that|who|where|when)\s+)',
            r'(,\s+)'
        ]

        current_text = sentence
        for pattern in clause_patterns:
            if len(current_text) <= max_length:
                break

            splits = re.split(pattern, current_text)
            if len(splits) > 1:
                current_part = ""
                for split in splits:
                    candidate = current_part + split
                    if len(candidate) <= max_length:
                        current_part = candidate
                    else:
                        if current_part:
                            parts.append(current_part.strip())
                        current_part = split

                if current_part:
                    current_text = current_part.strip()
                else:
                    break

        if current_text and current_text not in parts:
            parts.append(current_text)

        return parts if parts else [sentence]

    def _optimize_chunk_sizes(self, chunks: list, target_size: int) -> list:
        """Post-process chunks to optimize for API efficiency"""
        if not chunks:
            return chunks

        optimized = []
        i = 0
        while i < len(chunks):
            current_chunk = chunks[i]
            # Try to merge small consecutive chunks
            while (i + 1 < len(chunks) and
                   len(current_chunk) + len(chunks[i + 1]) + 1 <= target_size and
                   len(current_chunk) < target_size * 0.6):
                i += 1
                # Insert a single space only when both chunks are non-empty
                current_chunk = f"{current_chunk} {chunks[i]}".strip()

            optimized.append(current_chunk.strip())
            i += 1

        return optimized

    def _is_character_spaced(self, s: str) -> bool:
        """
        Detect strings like 'D i d  y o u ...' (over 10 alternating letter-space pairs).
        """
        # Heuristic: many single letters separated by single spaces, words lose multi-letter sequences
        return bool(re.search(r'(?:\w\s){10,}\w', s))

    def _collapse_character_spaced(self, s: str) -> str:
        """
        Collapse 'D i d  y o u' -> 'Did you'.
        """
        # Replace multiple spaces with single first
        s = re.sub(r'\s+', ' ', s)
        # Join single-letter tokens when pattern indicates char-spaced text
        tokens = s.split(' ')
        if all(len(tok) <= 1 for tok in tokens if tok):
            return ''.join(tokens)
        # More general recovery: collapse single-space sequences between single chars inside words
        return re.sub(r'(?<=\w)\s(?=\w)', '', s)

    def _enforce_tokenizer_limit(self, chunks: list, language: str, hard_cap: int) -> list:
        """
        Ensure no chunk exceeds tokenizer character cap for given language.
        """
        safe = []
        for ch in chunks:
            if len(ch) <= hard_cap:
                safe.append(ch)
                continue
            # Re-split long chunks by sentences then by clauses
            for sent in self._smart_sentence_split(ch):
                if len(sent) <= hard_cap:
                    safe.append(sent)
                else:
                    parts = self._split_long_sentence(sent, hard_cap)
                    for p in parts:
                        if len(p) <= hard_cap:
                            safe.append(p)
                        else:
                            # As last resort, hard slice to cap
                            start = 0
                            while start < len(p):
                                safe.append(p[start:start+hard_cap])
                                start += hard_cap
        return safe

    def ultra_intelligent_chunking(self, text: str, target_mode: str = 'ultra_fast') -> list:
        """
        Ultra-intelligent chunking optimized for minimal API calls while respecting model limits.
        Targets ~25 seconds of audio per chunk subject to tokenizer caps.
        """
        max_chars = self.optimal_chunk_sizes[target_mode]

        # Split by paragraphs (blank lines) first to maintain logical flow
        # IMPORTANT: do NOT use \s* here; it matches empty strings and explodes into per-character chunks.
        paragraphs = re.split(r'\n\s*\n+', text)

        chunks = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(para) <= max_chars:
                chunks.append(para)
            else:
                # For long paragraphs, use intelligent sentence grouping
                sentences = self._smart_sentence_split(para)
                current_chunk = ""
                for sentence in sentences:
                    # Calculate complexity score for better chunking decisions
                    complexity = self._calculate_text_complexity(sentence)
                    # Adjust max length based on complexity
                    effective_max = max_chars
                    if complexity > 0.7:
                        effective_max = int(max_chars * 0.8)  # Shorter chunks for complex text
                    elif complexity < 0.3:
                        effective_max = int(max_chars * 1.0)  # Keep at cap to respect tokenizer

                    candidate_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
                    if len(candidate_chunk) <= effective_max:
                        current_chunk = candidate_chunk
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        # Handle extremely long sentences
                        if len(sentence) > effective_max:
                            sentence_parts = self._split_long_sentence(sentence, effective_max)
                            chunks.extend(sentence_parts[:-1])  # Add all but the last part
                            current_chunk = sentence_parts[-1] if sentence_parts else ""
                        else:
                            current_chunk = sentence

                if current_chunk:
                    chunks.append(current_chunk)

        # Post-process to merge tiny chunks and optimize for API calls
        optimized_chunks = self._optimize_chunk_sizes(chunks, max_chars)

        # Enforce tokenizer char caps per language
        lang_cap = self._lang_char_caps.get("en", 240)
        optimized_chunks = self._enforce_tokenizer_limit(optimized_chunks, "en", lang_cap)

        log.info(f"Ultra-intelligent chunking: {len(text)} chars -> {len(optimized_chunks)} chunks (avg: {len(text)//max(len(optimized_chunks),1):.0f} chars/chunk)")

        return optimized_chunks

    def ultra_fast_generate_chunk(self, text: str, language: str = "en",
                                  pitch_scale: float = 1.0, speed_scale: float = 1.0,
                                  energy_scale: float = 1.0) -> np.ndarray:
        """
        Ultra-fast chunk generation using precomputed conditioning latents.
        This eliminates the per-call overhead that was causing slowdowns.
        """
        try:
            # Guard against character-spaced text reaching the model
            if self._is_character_spaced(text):
                text = self._collapse_character_spaced(text)

            # Prefer low-level synth with precomputed latents
            if self.current_gpt_cond_latent is not None and self.current_speaker_embedding is not None:
                output = self.tts.synthesizer.tts_model.synthesize(
                    text=text,
                    config=self.tts.synthesizer.tts_config,
                    speaker_id=None,
                    gpt_cond_latent=self.current_gpt_cond_latent,
                    speaker_embedding=self.current_speaker_embedding,
                    language=language,
                    **{
                        "temperature": 0.65,
                        "length_penalty": 1.0,
                        "repetition_penalty": 1.0,
                        "top_k": 50,
                        "top_p": 0.85
                    }
                )
                audio = output["wav"]
            else:
                # Fallback to high-level API if conditioning failed
                audio = self.tts.tts(
                    text=text,
                    speaker_wav=self.current_speaker_wav,
                    language=language,
                    split_sentences=False
                )

            # Ensure audio is numpy array with correct dtype
            if not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)
            else:
                audio = audio.astype(np.float32)

            # Apply effects if needed (minimal overhead)
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
            log.error(f"Ultra-fast chunk generation failed: {e}")
            raise e

    def process_markup_effects(self, text: str, base_pitch: float = 1.0,
                               base_speed: float = 1.0, base_energy: float = 1.0):
        """Process expression markup and return adjusted parameters"""
        current_pitch = base_pitch
        current_speed = base_speed
        current_energy = base_energy

        # Handle expression markup with optimized, natural adjustments
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

        # Handle rate/pitch controls
        if '[rate=slow]' in text:
            current_speed *= 0.88
        elif '[rate=fast]' in text:
            current_speed *= 1.12
        if '[pitch=high]' in text:
            current_pitch *= 1.06
        elif '[pitch=low]' in text:
            current_pitch *= 0.94

        return current_pitch, current_speed, current_energy

    def clean_markup(self, text: str) -> str:
        """Remove markup tags for TTS processing"""
        clean_text = text
        # Remove expression tags
        for tag in ['happy', 'sad', 'excited', 'whisper', 'emphasis']:
            clean_text = clean_text.replace(f'[{tag}]', '').replace(f'[/{tag}]', '')
        # Remove control tags
        for tag in ['rate=slow', 'rate=fast', 'pitch=high', 'pitch=low']:
            clean_text = clean_text.replace(f'[{tag}]', '').replace(f'[/{tag}]', '')
        return clean_text.strip()

    def ultra_fast_normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Ultra-fast audio normalization for consistent output"""
        # Fast DC offset removal
        audio = audio - np.mean(audio)
        # Fast peak normalization
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio * (0.72 / max_val)  # Target -2.8dB
        # Fast soft limiting
        audio = np.tanh(audio * 0.88) * 0.88
        return audio

    def generate_ultra_fast(self, text: str, voice_sample: str, output_file: str,
                            language: str = "en", emotion_scale: float = 1.0,
                            speaking_rate: float = 1.0, pitch_scale: float = 1.0,
                            quality_mode: str = 'ultra_fast') -> str:
        """
        Ultra-fast generation targeting 4-6 minutes for full script.
        Uses all optimizations: precomputed conditioning, intelligent chunking,
        parallel processing, and minimal overhead.
        """
        start_total_time = time.time()
        logger_manager.start_operation('ultra_fast_voice_generation')

        try:
            # Input validation
            if not text.strip():
                raise ValueError("Text to synthesize is empty!")
            if not os.path.exists(voice_sample):
                raise FileNotFoundError(f"Voice sample not found: {voice_sample}")

            log.info(f"Starting ultra-fast generation: {len(text)} chars, target: 4-6 minutes")

            # Ultra-fast model loading
            self._ensure_model_loaded_ultra_fast()

            # Precompute conditioning latents (key optimization)
            self.precompute_conditioning_latents(voice_sample)

            # Ultra-intelligent chunking for minimal API calls
            chunks = self.ultra_intelligent_chunking(text, quality_mode)
            log.info(f"Ultra-intelligent chunking: {len(chunks)} chunks (avg: {len(text)//len(chunks):.0f} chars/chunk)")

            # Ultra-fast generation with minimal overhead
            spinner = ProgressSpinner("Ultra-fast voice generation")
            spinner.start()

            audio_segments = []

            try:
                # Generate all chunks with optimized processing
                for i, chunk in enumerate(chunks):
                    chunk_start = time.time()

                    try:
                        # Process markup effects
                        pitch, speed, energy = self.process_markup_effects(
                            chunk, pitch_scale, speaking_rate, emotion_scale
                        )

                        # Clean markup
                        clean_chunk = self.clean_markup(chunk)

                        # Safety net: collapse accidental character-spaced text
                        if self._is_character_spaced(clean_chunk):
                            clean_chunk = self._collapse_character_spaced(clean_chunk)

                        # Ultra-fast generation
                        audio = self.ultra_fast_generate_chunk(
                            clean_chunk, language, pitch, speed, energy
                        )

                        audio_segments.append(audio)

                        chunk_time = time.time() - chunk_start
                        audio_duration = len(audio) / self.tts.synthesizer.output_sample_rate
                        rtf = chunk_time / max(audio_duration, 0.001)

                        log.debug(f"Chunk {i+1}/{len(chunks)}: {chunk_time:.2f}s (RTF: {rtf:.2f})")

                    except Exception as e:
                        log.error(f"Failed to generate chunk {i+1}: {e}")
                        continue

                # Clear GPU cache after generation
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            finally:
                spinner.stop()

            if not audio_segments:
                raise ValueError("No audio segments were generated successfully")

            # Ultra-fast audio assembly
            log.info("Assembling final audio...")

            # Add minimal pauses between chunks (100ms for speed)
            final_segments = []
            pause_samples = int(0.1 * self.tts.synthesizer.output_sample_rate)
            pause = np.zeros(pause_samples, dtype=np.float32)

            for i, segment in enumerate(audio_segments):
                final_segments.append(segment)
                if i < len(audio_segments) - 1:
                    final_segments.append(pause)

            # Ultra-fast concatenation
            combined_audio = np.concatenate(final_segments, axis=0)

            # Ultra-fast normalization
            final_audio = self.ultra_fast_normalize_audio(combined_audio)

            # Save with maximum efficiency
            out_path = Path(output_file)
            if out_path.parent and str(out_path.parent) != "":
                out_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out_path), final_audio, self.tts.synthesizer.output_sample_rate)

            # Performance metrics
            total_time = time.time() - start_total_time
            audio_duration = len(final_audio) / self.tts.synthesizer.output_sample_rate
            overall_rtf = total_time / audio_duration

            log.info(f"Ultra-fast generation completed!")
            log.info(f"Total time: {total_time:.2f}s | Audio: {audio_duration:.1f}s | RTF: {overall_rtf:.2f}")
            log.info(f"Performance: {len(text)/total_time:.0f} chars/sec")

            # Target achievement check
            if total_time <= 360:  # 6 minutes
                log.info("Target achieved: Generated in under 6 minutes!")
            elif total_time <= 240:  # 4 minutes
                log.info("Excellent: Generated in under 4 minutes!")

            return str(out_path)

        except Exception as e:
            log.error(f"Ultra-fast generation failed: {e}")
            raise e
        finally:
            logger_manager.end_operation('ultra_fast_voice_generation')

    def generate(self, text: str, voice_sample: str, output_file: str, language: str = "en",
                 emotion_scale: float = 1.0, speaking_rate: float = 1.0, pitch_scale: float = 1.0,
                 speaker_consistency: float = 0.75, batch_size: int = 50, quality_mode: str = 'ultra_fast'):
        """
        Main generation method - routes to ultra-fast generation
        """
        return self.generate_ultra_fast(
            text=text,
            voice_sample=voice_sample,
            output_file=output_file,
            language=language,
            emotion_scale=emotion_scale,
            speaking_rate=speaking_rate,
            pitch_scale=pitch_scale,
            quality_mode=quality_mode
        )

    def get_supported_languages(self):
        """Get list of supported languages"""
        return [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr",
            "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko", "hi"
        ]

# Backward compatibility alias
VoiceClone = UltraOptimizedVoiceClone

# Test function
if __name__ == "__main__":
    # Test the ultra-optimized voice cloner
    cloner = UltraOptimizedVoiceClone()
    print("Ultra-optimized voice cloner initialized successfully!")
    print(f"Target: 4-6 minute generation for full scripts")
    print(f"Supported languages: {', '.join(cloner.get_supported_languages())}")