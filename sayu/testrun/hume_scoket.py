import base64
import datetime
from hume import  Stream 
from hume.empathic_voice.chat.types import SubscribeEvent

class WebSocketHandler:
    def __init__(self):
        self.byte_strs = Stream.new()

    async def on_open(self):
        print("WebSocket connection opened.")

    async def on_message(self, message: SubscribeEvent):
        if message.type == "chat_metadata":
            self._print_prompt(f"<{message.type}> Chat ID: {message.chat_id}, Chat Group ID: {message.chat_group_id}")
            return
        elif message.type == "user_message" or message.type == "assistant_message":
            self._print_prompt(f"{message.message.role}: {message.message.content}")
            if message.models.prosody is not None:
                self._print_emotion_scores(
                    self._extract_top_n_emotions(dict(message.models.prosody.scores), 3)
                )
            else:
                print("Emotion scores not available.")
            return
        elif message.type == "audio_output":
            await self.byte_strs.put(
                base64.b64decode(message.data.encode("utf-8"))
            )
            return
        elif message.type == "error":
            raise RuntimeError(f"Received error message from Hume websocket ({message.code}): {message.message}")
        else:
            self._print_prompt(f"<{message.type}>")
    
    async def on_close(self):
        print("WebSocket connection closed.")

    async def on_error(self, error):
        print(f"Error: {error}")

    def _print_prompt(self, text: str) -> None:
        now = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%H:%M:%S")
        print(f"[{now}] {text}")

    def _extract_top_n_emotions(self, emotion_scores: dict, n: int) -> dict:
        sorted_emotions = sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)
        top_n_emotions = {emotion: score for emotion, score in sorted_emotions[:n]}

        return top_n_emotions

    def _print_emotion_scores(self, emotion_scores: dict) -> None:
        print(
            ' | '.join([f"{emotion} ({score:.2f})" for emotion, score in emotion_scores.items()])
        )
