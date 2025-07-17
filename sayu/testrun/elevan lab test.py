import io
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
from pydub.playback import play

# Initialize the ElevenLabs client
# It will automatically pick up your ELEVEN_API_KEY from environment variables
# or you can pass it directly: client = ElevenLabs(api_key="YOUR_API_KEY")
e_apikey = 'sk_9a6651c969660fb533cbe382e33f1b9ede8317ef6a2375db'

client = ElevenLabs(api_key=e_apikey)

text_to_convert = "This is a demonstration of converting MP3 to WAV for pydub playback without ffplay."
voice_id_to_use = "YOUR_VOICE_ID" # Replace with a real voice ID or voice name like "Rachel"

try:
    # 1. Convert text to speech using ElevenLabs API, requesting MP3
    audio_bytes_mp3 = client.text_to_speech.convert(
        text=text_to_convert,
        voice_id=voice_id_to_use,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128" # Request MP3
    )

    # 2. Load the MP3 bytes into a pydub AudioSegment
    audio_stream_mp3 = io.BytesIO(audio_bytes_mp3)
    audio_segment_mp3 = AudioSegment.from_file(audio_stream_mp3, format="mp3")

    # 3. Play the AudioSegment (pydub will use simpleaudio if installed for WAV,
    # or rely on other backends for MP3 if available, e.g., FFmpeg)
    # For MP3 playback without FFmpeg, simpleaudio needs to be able to handle it,
    # or pydub needs another backend.
    # To *guarantee* simpleaudio for playback, export to WAV first:
    audio_segment_wav = audio_segment_mp3.export(format="wav")
    
    # Wrap the WAV bytes in a BytesIO object for play
    audio_stream_wav = io.BytesIO(audio_segment_wav.read())

    # Load the WAV stream into a new AudioSegment (or directly play audio_segment_mp3 if simpleaudio handles MP3)
    final_audio_segment = AudioSegment.from_file(audio_stream_wav, format="wav")


    print("Playing audio with pydub (after converting to WAV for simpleaudio)...")
    play(final_audio_segment)
    print("Audio playback finished.")

except Exception as e:
    print(f"An error occurred: {e}")
    print("If you're still seeing FFmpeg related errors, it means pydub's MP3 handling or your")
    print("'simpleaudio' setup might still implicitly require FFmpeg for MP3 decoding.")
    print("Consider using 'output_format=\"pcm_44100\"' directly from ElevenLabs for minimal dependencies.")