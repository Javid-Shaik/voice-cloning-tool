"""
Text Processor for Voice Cloning
Optimizes text for natural speech synthesis
"""

def process_text(raw_text: str) -> str:
    """
    Enhance text for natural speech synthesis:
    - Add proper spacing after punctuation
    - Split overly long sentences
    - Clean up formatting issues
    - Handle abbreviations and numbers
    
    Args:
        raw_text (str): Original text to process
        
    Returns:
        str: Processed text optimized for TTS
    """
    if not raw_text or not raw_text.strip():
        return ""
    
    # Basic cleanup
    text = raw_text.strip()
    
    # Normalize line breaks and excessive whitespace
    text = ' '.join(text.split())
    
    # Fix punctuation spacing
    text = text.replace("...", "…")  # Replace triple dots with ellipsis
    text = text.replace(".", ". ")
    text = text.replace("?", "? ")
    text = text.replace("!", "! ")
    text = text.replace(",", ", ")
    text = text.replace(";", "; ")
    text = text.replace(":", ": ")
    
    # Fix double spaces created by replacements
    while "  " in text:
        text = text.replace("  ", " ")
    
    # Split sentences that are too long (better for TTS processing)
    sentences = text.split(". ")
    enhanced_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If sentence is very long, try to split at natural break points
        if len(sentence.split()) > 20:  # More than 20 words
            # Try splitting at commas, semicolons, or conjunctions
            parts = []
            for part in sentence.split(", "):
                if len(part.split()) > 15:
                    # Further split on "and", "but", "or", "so"
                    for conjunction in [" and ", " but ", " or ", " so ", " because "]:
                        if conjunction in part:
                            sub_parts = part.split(conjunction)
                            if len(sub_parts) == 2:
                                parts.append(sub_parts[0].strip())
                                parts.append(conjunction.strip() + " " + sub_parts[1].strip())
                                break
                    else:
                        parts.append(part.strip())
                else:
                    parts.append(part.strip())
            enhanced_sentences.extend([p for p in parts if p])
        else:
            enhanced_sentences.append(sentence)
    
    # Join sentences back together
    result = ". ".join([s.strip() for s in enhanced_sentences if s.strip()])
    
    # Final cleanup
    result = result.replace("…", "...")  # Convert back to triple dots
    result = result.replace(" .", ".")
    result = result.replace(" ?", "?")
    result = result.replace(" !", "!")
    result = result.replace(" ,", ",")
    
    # Ensure proper sentence ending
    if result and not result.endswith(('.', '!', '?')):
        result += "."
    
    return result

# Test function
if __name__ == "__main__":
    # Test the text processor
    sample_text = """
    Hello!Welcome to my Python voice cloning demo.In just a few minutes,we will generate speech in my own voice.
    This is a very long sentence that probably should be broken up into smaller parts because it contains too many words and ideas that could be separated for better speech synthesis and more natural sounding output.
    Python is amazing   !  With this tool,you can convert any text into spoken audio.Let's have fun!
    """
    
    processed = process_text(sample_text)
    print("Original text:")
    print(repr(sample_text))
    print("\nProcessed text:")
    print(repr(processed))
    print("\nProcessed text (readable):")
    print(processed)