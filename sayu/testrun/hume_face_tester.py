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


#hume key variables 
hume_key = dcon("hume")
hume_secret = dcon("hume_secret")
hume_ai_Config = dcon("hume_config")

#load hume client
client = AsyncHumeClient(api_key=hume_key)
model_config = Config(face=StreamFace())


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

def main():
    cap = cv2.VideoCapture(0) #set to 0 for default device webcam
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

if __name__ == "__main__":
    main()