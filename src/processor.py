"""
Text Processor for Voice Cloning
Optimizes text for natural speech synthesis
"""
import re
from utils.logging import get_logger

log = get_logger('TextProcessor')

def process_text(raw_text: str) -> str:
    """
    Enhanced text processing for expressive speech synthesis.
    This version returns a cleaned string without SSML, as the `VoiceClone` class
    now handles the markup and applies the effects after generation.
    """
    if not raw_text or not raw_text.strip():
        log.warning("Received empty text for processing.")
        return ""
    
    # Normalize line breaks
    text = raw_text.strip()
    text = re.sub(r'\s*\n\s*', ' ', text)
    
    # Fix punctuation spacing
    text = re.sub(r'([.?!,;:])\s*', r'\1 ', text)
    text = re.sub(r'\s{2,}', ' ', text) # Remove double spaces
    
    # Simple and fast sentence splitting
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[.?!])\s+', text)
    
    # Join sentences back together with a single space to avoid
    # a long single string which causes the tokenization error
    processed_text = ' '.join(sentences)
    
    log.debug(f"Original text length: {len(raw_text)} -> Processed text length: {len(processed_text)}")
    return processed_text.strip()

# Test function
if __name__ == "__main__":
    # Test the text processor
    sample_text = """
    Hello! Welcome to my Python voice cloning demo. In just a few minutes, we will generate speech in my own voice.
    This is a very long sentence that probably should be broken up into smaller parts because it contains too many words and ideas that could be separated for better speech synthesis and more natural sounding output.
    Python is amazing ! With this tool, you can convert any text into spoken audio. Let's have fun!
    """
    
    processed = process_text(sample_text)
    print("Original text:")
    print(repr(sample_text))
    print("\nProcessed text:")
    print(repr(processed))
    print("\nProcessed text (readable):")
    print(processed)