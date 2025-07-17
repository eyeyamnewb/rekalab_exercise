import asyncio
import cv2

from decouple import config
from hume import AsyncHumeClient
from hume.expression_measurement.stream import Config
from hume.expression_measurement.stream.socket_client import StreamConnectOptions
from hume.expression_measurement.stream.stream.types.stream_face import StreamFace
import numpy as np # For OpenCV frame handling
import io # To convert numpy array to bytes

# Replace with your actual API key
HUME_API_KEY = config("hume")

async def main():
    client = AsyncHumeClient(api_key=HUME_API_KEY)
    model_config = Config(face=StreamFace())
    stream_options = StreamConnectOptions(config=model_config)

    # Initialize OpenCV for webcam
    cap = cv2.VideoCapture(1) # 0 for default webcam, change if needed

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    async with client.expression_measurement.stream.connect(options=stream_options) as socket:
        print("Connected to Hume stream. Press 'q' to quit.")
        while True:
            ret, frame = cap.read() # Read a frame from the webcam

            if not ret:
                print("Failed to grab frame.")
                break

            # Convert OpenCV frame (NumPy array) to JPEG bytes
            # You might need to experiment with compression quality for performance
            is_success, buffer = cv2.imencode(".jpg", frame)
            if not is_success:
                print("Failed to encode frame.")
                continue

            frame_bytes = io.BytesIO(buffer).read()

            try:
                # Send the frame bytes to Hume
                # Hume's streaming API might have a specific method for sending bytes,
                # you'll need to consult the Hume SDK documentation for the exact method name.
                # It's likely something like socket.send_bytes() or socket.send_image().
                # For now, let's assume `send_bytes` or similar is available for stream input
                result = await socket.send_bytes(frame_bytes) # THIS IS A PLACEHOLDER - CHECK HUME DOCS

                # Process Hume's results
                if result and result.results:
                    for prediction in result.results.predictions:
                        # Access face data and expressions
                        for face in prediction.models.face.grouped_predictions:
                            emotions = face.predictions[0].emotions # Assuming single face
                            for emotion in emotions:
                                # You can print or overlay these emotions on the frame
                                print(f"Emotion: {emotion.name}, Score: {emotion.score:.2f}")

                # Optional: Display the frame with OpenCV (you can add Hume results here)
                cv2.imshow('Live Feed with Hume Analysis', frame)

            except Exception as e:
                print(f"Error sending frame to Hume: {e}")
                # Depending on the error, you might want to break or continue

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release OpenCV resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())