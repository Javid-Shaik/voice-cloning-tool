# text_encoding.py
"""
Text Encoding and Segmentation Module with Hierarchical Prosody
- Converts raw or SSML-processed text into phoneme and word-level tokens
- Generates alignment mapping for prosody models (duration, pitch, energy predictors)
- Implements hierarchical prosody modeling with transformer-based architecture
"""

import re
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
import phonemizer
from phonemizer.backend import EspeakBackend
from transformers import AutoTokenizer, AutoModel
from processor import process_text
from ssml_processor import SSMLProcessor

class HierarchicalProsodyModel(nn.Module):
    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Phoneme-level components
        self.phoneme_duration = nn.Linear(hidden_size, 1)
        self.phoneme_pitch = nn.Linear(hidden_size, 1)
        self.phoneme_energy = nn.Linear(hidden_size, 1)
        
        # Word-level components
        self.word_stress = nn.Linear(hidden_size, 1)
        self.word_emphasis = nn.Linear(hidden_size, 1)
        
        # Sentence-level components
        self.sentence_pitch = nn.Linear(hidden_size, 1)
        self.sentence_rate = nn.Linear(hidden_size, 1)
        
    def forward(self, 
                phoneme_features: torch.Tensor,
                word_features: torch.Tensor,
                sentence_features: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        # Phoneme-level predictions
        durations = self.phoneme_duration(phoneme_features).squeeze(-1)
        phoneme_pitches = self.phoneme_pitch(phoneme_features).squeeze(-1)
        energies = self.phoneme_energy(phoneme_features).squeeze(-1)
        
        # Word-level predictions
        stress = self.word_stress(word_features).squeeze(-1)
        emphasis = self.word_emphasis(word_features).squeeze(-1)
        
        # Sentence-level predictions
        sent_pitch = self.sentence_pitch(sentence_features).squeeze(-1)
        speaking_rate = self.sentence_rate(sentence_features).squeeze(-1)
        
        return (durations, phoneme_pitches, energies, 
                stress, emphasis,
                sent_pitch, speaking_rate)

class TransformerEncoder(nn.Module):
    def __init__(self, layers: int = 12, heads: int = 16, hidden_size: int = 768):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            batch_first=True
        )
        self.prosody_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )
        
        self.prosody_model = HierarchicalProsodyModel(hidden_size=hidden_size)

class TextEncoder:
    """
    Encodes input text (plain or SSML) into phoneme and word tokens with alignment info.
    """
    def __init__(self, language: str = 'en-us'):
        # Initialize phonemizer backend for phoneme conversion
        self.phonemizer = phonemizer.Phonemizer(
            backend=EspeakBackend(language=language),
            preserve_punctuation=True,
            punctuation_marks=".,!?;:",
            with_stress=False
        )
        self.ssml = SSMLProcessor()

    def encode(self, raw_input: str) -> Dict[str, List]:
        """
        Args:
            raw_input: plain text or SSML string
        Returns:
            {
                'words': List[str],
                'phonemes': List[str],
                'word_to_phoneme': List[Tuple[int,int]],  # (start, end) indices in phoneme list
                'ssml_segments': List[Dict]  # parsed SSML for advanced prosody
            }
        """
        # Detect and parse SSML if present
        if '<' in raw_input and '>' in raw_input:
            ssml_segments = self.ssml.parse(raw_input)
            # Flatten SSML text for phoneme conversion
            flat_text = self.ssml.flatten_segments_to_text(ssml_segments)
        else:
            ssml_segments = []
            flat_text = process_text(raw_input)

        # Split into words
        words = re.findall(r"\b\w[\w']*\b|[.,!?;:]", flat_text)

        # Convert to phonemes (joined string)
        phoneme_str = self.phonemizer.phonemize(flat_text, strip=True)
        # phonemizer returns strings like "HH AH L OW"
        phonemes = phoneme_str.split()

        # Align words to phonemes by re-phonemizing each word
        word_to_phoneme = []
        phoneme_index = 0
        for word in words:
            # Skip punctuation
            if re.match(r'[.,!?;:]', word):
                word_to_phoneme.append((phoneme_index, phoneme_index))
                continue

            w_phonemes = self.phonemizer.phonemize(word, strip=True).split()
            start = phoneme_index
            end = phoneme_index + len(w_phonemes)
            word_to_phoneme.append((start, end))
            phoneme_index = end

        return {
            'words': words,
            'phonemes': phonemes,
            'word_to_phoneme': word_to_phoneme,
            'ssml_segments': ssml_segments
        }

# Chunking utility aligned with prosody frame counts
def chunk_for_prosody(tokens: List[str], max_tokens: int = 50) -> List[List[str]]:
    """
    Splits token list into chunks of up to max_tokens, preserving word boundaries.
    Returns list of token chunks.
    """
    chunks = []
    current = []
    for token in tokens:
        if len(current) >= max_tokens:
            chunks.append(current)
            current = []
        current.append(token)
    if current:
        chunks.append(current)
    return chunks

# Example alignment to frames

def align_to_frames(durations: List[float], frame_shift_ms: float = 10.0) -> List[int]:
    """
    Convert predicted durations (in ms) per token to frame counts.
    frame_shift_ms: e.g., 10ms per frame
    Returns list of frame counts per token.
    """
    frames = []
    for dur in durations:
        frame_count = max(1, int(round(dur / frame_shift_ms)))
        frames.append(frame_count)
    return frames