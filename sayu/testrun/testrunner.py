import whisper
import pyaudio
import os
import io
from pydub import AudioSegment
#from pydub.playback import play
from elevenlabs.client import ElevenLabs
from elevenlabs import play, stream
import elevenlabs as elly
import numpy as np
import pyttsx3
import transformers

from gtts import gTTS #incase i ran out of monthly token key for elevenlab

e_apikey = 'sk_9a6651c969660fb533cbe382e33f1b9ede8317ef6a2375db'
whisp_model = whisper.load_model('medium')
client = ElevenLabs(api_key=e_apikey)

sample_voice_path = '../sayu_emotion_sample'
#DICTIONARY FOR MAPPING 
mood_dictionary = {
                'happy': 1,'sad': 0,
                'neutral':2,'anxious':4,
                'depression':5 ,'angry':3, 
                   }

def emotion_sampling():
    pass

# Audio recording configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5  # Duration to record (seconds)

#function listen to audio 
def record_audio():
    p = pyaudio.PyAudio()
    print("Listening...")
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    # Convert bytes to numpy array (float32 for Whisper)
    audio_data = b''.join(frames)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    return audio_np

#function to trnanscribe words to text 
def transcribe(audio_np, language='en'):
    result = whisp_model.transcribe(audio_np, language=language, fp16=False)
    return result
    
#mini ai logic

def tiny_brain(words):
    text = words.lower().strip()
    if 'hello' in text:
        response = "hello there"
    elif 'how are you' in text:
        response = "i'm fine"
    else:
        response = "I did not understand that."
    
    return response
    #print(response)
    #tts = gTTS(response)
    #fp = io.BytesIO()
    #tts.write_to_fp(fp)
    #fp.seek(0)
    #audio = AudioSegment.from_file(fp, format="mp3")
    #play(audio)
    #os.system("start response.mp3")  # On Windows; use "afplay" on Mac or "mpg123" on Linux

if __name__ == '__main__':
    try:
        while True:
            audio_np = record_audio()
            result = transcribe(audio_np)
            print(f"heard: {result.get('text', '')}")
            print('-' * 50)
            text = tiny_brain(result.get('text', ''))
            
            try:
                audio = client.text_to_speech.stream(
                    text=text,
                    voice_id="JBFqnCBsd6RMkjVDRZzb",
                    model_id="eleven_multilingual_v2",
                    output_format="mp3_44100_128",
                )
                stream(audio)
            except Exception as e:
                print("TTS error:", e)
    except KeyboardInterrupt:
        print("\nExiting...")