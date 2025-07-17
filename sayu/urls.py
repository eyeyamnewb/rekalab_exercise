
from django.urls import path , include
from. import views

urlpatterns = [
   path("", views.home, name="home"),
   path("initialize-ai/", views.initialize_ai, name="initialize_ai"),
   path("analyze-emotion/", views.analyze_emotion, name="analyze_emotion"),
   path("process-audio/", views.process_audio, name="process_audio"),
   path("stream-audio-chunk/", views.stream_audio_chunk, name="stream_audio_chunk"),
   path("poll-ai-response/", views.poll_ai_response, name="poll_ai_response"),
   path("stop-ai-session/", views.stop_ai_session, name="stop_ai_session"),
]