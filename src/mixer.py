from pydub import AudioSegment

def mix_with_music(voice_file, music_file, output_file, music_volume=-15):
    """
    Mix TTS voice with background music.
    """
    voice = AudioSegment.from_wav(voice_file)
    music = AudioSegment.from_mp3(music_file) - abs(music_volume)

    # Loop music to match voice length
    music = music[:len(voice)]
    mixed = voice.overlay(music)

    mixed.export(output_file, format="wav")
    return output_file
