import asyncio
import cv2
import mediapipe as mp
import tempfile
import os
from decouple import config as dcon
from hume import AsyncHumeClient
from hume.expression_measurement.stream import Config
from hume.expression_measurement.stream.socket_client import StreamConnectOptions
from hume.expression_measurement.stream.stream.types.stream_face import StreamFace
from hume.empathic_voice.chat.socket_client import ChatConnectOptions
from hume.empathic_voice.chat.types import SubscribeEvent
from hume import HumeClient, MicrophoneInterface , Stream 
from hume_scoket import WebSocketHandler    

from hume.expression_measurement.stream.stream.types.config import Config
from hume.expression_measurement.stream.stream.types.stream_language import StreamLanguage
from hume.expression_measurement.stream.stream.types.stream_model_predictions import StreamModelPredictions
from hume.expression_measurement.stream.stream.types.subscribe_event import SubscribeEvent
from hume.expression_measurement.stream.socket_client import StreamConnectOptions
from hume.expression_measurement.stream.stream.types.stream_face import StreamFace
import numpy as np # For OpenCV frame handling
import io # To convert numpy array to bytes


"""# pyttsesx3 init 
engine = pyttsx3.init()
# hume init
client = AsyncHumeClient(api_key=config('hume'))
client.empathic_voice.configs.list_configs()

whisp_model = whisper.load_model("medium")  # Load Whisper model

sample_voice_path = '../sayu_emotion_sample'

emotion_data_path_Dict= {
    'angry': 'emotion_ogg/angry',
    'happy': 'emotion_ogg/happy',
    'sad': 'emotion_ogg/sad',       #considere tese section next time
    'nervous':'emotion_ogg/nervous',
    'excite':'emotion_ogg/excite',
}


# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050  # Librosa default sample rate
RECORD_SECONDS = 5

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

def transcribe(audio_np, language='en'):
    result = whisp_model.transcribe(audio_np, language=language, fp16=False)
    return result

transcribe_result = transcribe(record_audio())

client.tts.synthesize(
    text=transcribe_result['text'],
    voice='')

#engine.say(transcribe_result['text'])
#engine.runAndWait()"""

#hume key variables 
hume_key = dcon("hume")
hume_secret = dcon("hume_secret")
hume_ai_Config = dcon("hume_config")

#load hume client
client = AsyncHumeClient(api_key=hume_key)
model_config = Config(face=StreamFace())

client.empathic_voice.prompts.

async def voice_detect_or_talking_buddy() -> None:

    client = AsyncHumeClient(api_key=hume_key)
    options = ChatConnectOptions(config_id=hume_ai_Config, secret_key=hume_secret)

    websocket_handler = WebSocketHandler()

    async with client.empathic_voice.chat.connect_with_callbacks(
        options=options,
        on_open=websocket_handler.on_open,
        on_message=websocket_handler.on_message,
        on_close=websocket_handler.on_close,
        on_error=websocket_handler.on_error
    ) as socket:
        await asyncio.create_task(
            MicrophoneInterface.start(
                socket,
                allow_user_interrupt=False,
                byte_stream=websocket_handler.byte_strs
            )
        )

#frame annalyezer function 
async def analyze_frame(frame_bytes):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(frame_bytes)
        tmp_path = tmp.name

    stream_options = StreamConnectOptions(config=model_config)
    async with client.expression_measurement.stream.connect(options=stream_options) as socket:
        result = await socket.send_file(tmp_path)
        os.remove(tmp_path)
        predictions = getattr(result.face, "predictions", None)
        if not predictions:
            return []
        emo_names = [] # prepare list to collect emotions
        for r in predictions: # iterate through model scores predictions  for each emotion name
            emo_names.extend([(e.name, e.score) for e in r.emotions if e.score > 0.5])
        return emo_names

def facial_emotion_recognition():
    cap = cv2.VideoCapture(1) #set to 0 for default device webcam
    mp_face = mp.solutions.face_detection #congig mediapipe for dectecting faces
    face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    while True:
        ret, frame = cap.read() #read frame data from webcam
        if not ret:
            break

        # Detect faces with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #give output in color
        results = face_detection.process(rgb_frame) #mediapipe process the rgb frame to detect faces

        if results.detections:
            # Encode frame as JPEG
            ret, buf = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            emo_list = asyncio.run(analyze_frame(buf.tobytes())) 
            print("Detected emotions:", emo_list)
            # Optionally, draw face bounding boxes
            for detection in results.detections: 
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            print("No face detected locally.")

        cv2.imshow('Webcam', frame) # show video feed 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

async def facial_emotion_recognition_async():
    loop = asyncio.get_event_loop()
    def run_cv():
        facial_emotion_recognition()  # your existing function
    await loop.run_in_executor(None, run_cv)

async def main():
    await asyncio.gather(
        voice_detect_or_talking_buddy(),
        facial_emotion_recognition_async()
    )

if __name__ == "__main__":
    asyncio.run(main())