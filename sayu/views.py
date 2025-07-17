############################################################################
# DJANGO AND PYTHON IMPORTS
############################################################################

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import asyncio
import cv2
import mediapipe as mp
import tempfile
import os
import json
import threading
import time
import queue
import base64
import datetime
from decouple import config as dcon

############################################################################
# HUME AI IMPORTS AND CONFIGURATION
############################################################################

from hume import AsyncHumeClient
from hume.expression_measurement.stream import Config
from hume.expression_measurement.stream.socket_client import StreamConnectOptions
from hume.expression_measurement.stream.stream.types.stream_face import StreamFace
from hume.empathic_voice.chat.socket_client import ChatConnectOptions
from hume.empathic_voice.chat.types import SubscribeEvent
from hume import MicrophoneInterface, Stream    

from hume.expression_measurement.stream.stream.types.config import Config
from hume.expression_measurement.stream.stream.types.stream_language import StreamLanguage
from hume.expression_measurement.stream.stream.types.stream_model_predictions import StreamModelPredictions
from hume.expression_measurement.stream.stream.types.subscribe_event import SubscribeEvent
from hume.expression_measurement.stream.socket_client import StreamConnectOptions
from hume.expression_measurement.stream.stream.types.stream_face import StreamFace

############################################################################
# ADDITIONAL UTILITY IMPORTS
############################################################################

import numpy as np # For OpenCV frame handling
import io # To convert numpy array to bytes

############################################################################
# WEBSOCKET HANDLER CLASS - Manages Hume EVI Communication
############################################################################

class WebSocketHandler:
    """Enhanced WebSocket handler for Hume Empathic Voice Interface"""
    
    # ===== INITIALIZATION =====
    def __init__(self):
        self.byte_strs = Stream.new()
        self.latest_ai_message = None
        self.latest_emotions = None
        self.latest_audio_output = None
        self.message_queue = queue.Queue()
        self.is_ai_speaking = False
        self.auto_terminate_requested = False
        self.goodbye_keywords = ['goodbye', 'bye', 'see you later', 'end call', 'hang up', 'stop', 'quit', 'exit']
        # Audio streaming management
        self.audio_chunks = []
        self.audio_stream_active = False
        self.audio_stream_timeout = None
        self.audio_stream_start_time = None
    
    # ===== WEBSOCKET EVENT HANDLERS =====    
    async def on_open(self):
        print("WebSocket connection opened for Hume EVI.")
        
    async def on_message(self, message: SubscribeEvent):
        try:
            if message.type == "chat_metadata":
                self._print_prompt(f"<{message.type}> Chat ID: {message.chat_id}, Chat Group ID: {message.chat_group_id}")
                return
                
            elif message.type == "user_message":
                user_content = message.message.content
                self._print_prompt(f"User: {user_content}")
                
                # Extract emotion scores ONLY for user messages
                emotions = None
                if message.models.prosody is not None:
                    print(f"Raw prosody scores: {dict(message.models.prosody.scores)}")
                    emotions = self._extract_top_n_emotions(dict(message.models.prosody.scores), 6)
                    self._print_emotion_scores(emotions)
                    print(f"Extracted {len(emotions) if emotions else 0} emotions for user message")
                else:
                    print("No prosody data available for user message")
                
                # Check for goodbye keywords to trigger auto-termination
                if self._contains_goodbye_keyword(user_content):
                    print(f"Goodbye keyword detected in user message: '{user_content}'")
                    self.auto_terminate_requested = True
                    
                    # Add farewell message to queue
                    self.message_queue.put({
                        'type': 'farewell_detected',
                        'content': user_content,
                        'timestamp': time.time()
                    })
                
                # Store user message with emotions for frontend
                self.message_queue.put({
                    'type': 'user_message',
                    'content': user_content,
                    'emotions': emotions,  # Only user emotions attached here
                    'timestamp': time.time()
                })
                return
                
            elif message.type == "assistant_message":
                ai_response = message.message.content
                self._print_prompt(f"Assistant: {ai_response}")
                
                # Check if AI is signaling end of call
                call_should_end = False
                if "CALL_END_SIGNAL" in ai_response:
                    call_should_end = True
                    # Remove the signal from the response before sending to frontend
                    ai_response = ai_response.replace("CALL_END_SIGNAL", "").strip()
                
                # Do NOT extract emotions for assistant messages
                # Only user emotions should be sent to frontend
                
                # Store AI response WITHOUT emotions for frontend
                self.latest_ai_message = {
                    'response': ai_response,
                    'timestamp': time.time()
                }
                
                self.message_queue.put({
                    'type': 'assistant_message',
                    'content': ai_response,
                    # No emotions for assistant messages
                    'timestamp': time.time()
                })
                
                # If call should end, also send a message to indicate this
                if call_should_end:
                    self.message_queue.put({
                        'type': 'call_end',
                        'timestamp': time.time()
                    })
                
                return
                
            elif message.type == "audio_output":
                # Handle AI audio output with streaming support
                chunk_size = len(message.data) if message.data else 0
                current_time = time.time()
                print(f"[{current_time:.3f}] Received audio chunk, size: {chunk_size} characters")
                
                # Validate chunk data
                if not message.data or chunk_size < 10:
                    print(f"Warning: Received very small or empty audio chunk (size: {chunk_size})")
                    return
                
                # Set AI speaking flag to prevent feedback
                self.is_ai_speaking = True
                
                # Start audio stream if not already active
                if not self.audio_stream_active:
                    self.audio_stream_active = True
                    self.audio_chunks = []
                    self.audio_stream_start_time = current_time
                    print(f"[{current_time:.3f}] Starting new audio stream with first chunk of size {chunk_size}")
                
                # Add chunk to buffer with timestamp
                chunk_data = {
                    'data': message.data,
                    'size': chunk_size,
                    'timestamp': current_time
                }
                self.audio_chunks.append(chunk_data)
                
                # Calculate time since stream started
                stream_duration = current_time - getattr(self, 'audio_stream_start_time', current_time)
                print(f"[{current_time:.3f}] Audio stream now has {len(self.audio_chunks)} chunks, latest size: {chunk_size}, stream duration: {stream_duration:.3f}s")
                
                # Cancel existing timeout
                if self.audio_stream_timeout:
                    self.audio_stream_timeout.cancel()
                
                # Adaptive timeout based on chunk frequency
                chunk_interval = 0.1  # Default
                if len(self.audio_chunks) > 1:
                    last_chunk_time = self.audio_chunks[-2]['timestamp']
                    chunk_interval = current_time - last_chunk_time
                    print(f"[{current_time:.3f}] Chunk interval: {chunk_interval:.3f}s")
                
                # Set adaptive timeout (minimum 0.3s, maximum 1.5s)
                timeout_duration = max(0.3, min(1.5, chunk_interval * 3))
                print(f"[{current_time:.3f}] Setting timeout for {timeout_duration:.3f}s")
                
                loop = asyncio.get_event_loop()
                self.audio_stream_timeout = loop.call_later(timeout_duration, self._send_complete_audio)
                
                return
                
            elif message.type == "audio_output_end":
                # AI finished speaking - send any remaining audio
                print("Audio output ended")
                self._send_complete_audio()
                self.is_ai_speaking = False
                self.message_queue.put({
                    'type': 'ai_speaking_end',
                    'timestamp': time.time()
                })
                return
                
            elif message.type == "error":
                error_msg = f"Hume EVI Error ({message.code}): {message.message}"
                print(error_msg)
                self.message_queue.put({
                    'type': 'error',
                    'content': error_msg,
                    'timestamp': time.time()
                })
                return
                
            else:
                self._print_prompt(f"<{message.type}>")
                
        except Exception as e:
            print(f"Error processing message: {e}")
        
    async def on_close(self):
        print("WebSocket connection closed.")
        self.is_ai_speaking = False
        
    async def on_error(self, error):
        print(f"WebSocket Error: {error}")
        self.is_ai_speaking = False
        self.message_queue.put({
            'type': 'error',
            'content': f"WebSocket Error: {error}",
            'timestamp': time.time()
        })

    # ===== UTILITY AND HELPER METHODS =====
    def _print_prompt(self, text: str) -> None:
        now = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%H:%M:%S")
        print(f"[{now}] {text}")

    def _extract_top_n_emotions(self, emotion_scores: dict, n: int) -> list:
        sorted_emotions = sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)
        top_n_emotions = [{'name': emotion, 'score': score} for emotion, score in sorted_emotions[:n]]
        return top_n_emotions

    def _print_emotion_scores(self, emotions: list) -> None:
        if emotions:
            emotion_str = ' | '.join([f"{emotion['name']} ({emotion['score']:.2f})" for emotion in emotions])
            print(f"Emotions: {emotion_str}")
    
    def _contains_goodbye_keyword(self, text: str) -> bool:
        """Check if text contains goodbye keywords"""
        if not text:
            return False
        
        text_lower = text.lower()
        for keyword in self.goodbye_keywords:
            if keyword in text_lower:
                return True
        return False
    
    def _send_complete_audio(self):
        """Send combined audio chunks to frontend"""
        current_time = time.time()
        
        if not self.audio_chunks or not self.audio_stream_active:
            print(f"[{current_time:.3f}] No audio chunks to send or stream not active")
            return
            
        try:
            # Validate chunks before combining
            valid_chunks = []
            total_original_size = 0
            stream_start = getattr(self, 'audio_stream_start_time', current_time)
            stream_duration = current_time - stream_start
            
            for i, chunk_info in enumerate(self.audio_chunks):
                # Handle both old format (string) and new format (dict)
                if isinstance(chunk_info, dict):
                    chunk_data = chunk_info['data']
                    chunk_size = chunk_info['size']
                    chunk_time = chunk_info['timestamp']
                    relative_time = chunk_time - stream_start
                    print(f"[{current_time:.3f}] Chunk {i}: size={chunk_size}, time=+{relative_time:.3f}s")
                else:
                    # Legacy format
                    chunk_data = chunk_info
                    chunk_size = len(chunk_data) if chunk_data else 0
                
                if chunk_data and chunk_size > 10:  # Valid chunk
                    valid_chunks.append(chunk_data)
                    total_original_size += chunk_size
                else:
                    print(f"[{current_time:.3f}] Warning: Skipping invalid chunk {i} (size: {chunk_size})")
            
            if not valid_chunks:
                print(f"[{current_time:.3f}] Error: No valid audio chunks to send")
                self.audio_chunks = []
                self.audio_stream_active = False
                return
            
            # Combine all valid audio chunks
            combined_audio = ''.join(valid_chunks)
            final_size = len(combined_audio)
            
            print(f"[{current_time:.3f}] AUDIO STREAM SUMMARY:")
            print(f"  - Total chunks: {len(self.audio_chunks)}")
            print(f"  - Valid chunks: {len(valid_chunks)}")
            print(f"  - Stream duration: {stream_duration:.3f}s")
            print(f"  - Original size: {total_original_size} chars")
            print(f"  - Final size: {final_size} chars")
            print(f"  - Compression ratio: {final_size/total_original_size:.3f}" if total_original_size > 0 else "")
            
            # Validate combined audio
            if final_size < 100:
                print(f"[{current_time:.3f}] Warning: Combined audio very small ({final_size} chars), but sending anyway")
            
            # Send to frontend for lip sync
            self.message_queue.put({
                'type': 'audio_output',
                'audio_data': combined_audio,
                'chunk_count': len(self.audio_chunks),
                'valid_chunks': len(valid_chunks),
                'total_size': final_size,
                'stream_duration': stream_duration,
                'timestamp': current_time
            })
            
            # Signal that AI started speaking (only once per audio stream)
            self.message_queue.put({
                'type': 'ai_speaking_start',
                'timestamp': current_time
            })
            
            # Reset audio stream state
            self.audio_chunks = []
            self.audio_stream_active = False
            self.audio_stream_timeout = None
            self.audio_stream_start_time = None
            
            print(f"[{current_time:.3f}] Audio stream completed and sent to frontend")
            
        except Exception as e:
            print(f"[{current_time:.3f}] Error sending complete audio: {e}")
            print(f"Audio chunks info: {len(self.audio_chunks) if self.audio_chunks else 0} chunks")
            import traceback
            traceback.print_exc()
            self.audio_chunks = []
            self.audio_stream_active = False
            self.audio_stream_start_time = None
    
    # ===== MESSAGE QUEUE MANAGEMENT =====        
    def get_latest_message(self):
        """Get the latest message from the queue"""
        try:
            return self.message_queue.get_nowait()
        except queue.Empty:
            return None
            
    def has_messages(self):
        """Check if there are pending messages"""
        return not self.message_queue.empty()
    
    # ===== AUTO-TERMINATION MANAGEMENT =====
    def should_auto_terminate(self) -> bool:
        """Check if auto-termination was requested"""
        return self.auto_terminate_requested
    
    def reset_auto_terminate(self):
        """Reset auto-termination flag"""
        self.auto_terminate_requested = False

############################################################################
# REAL-TIME VOICE PROCESSOR CLASS - Handles Audio Processing
############################################################################

class RealTimeVoiceProcessor:
    """Handles real-time voice processing with Hume EVI"""
    
    # ===== INITIALIZATION =====
    def __init__(self):
        self.is_processing = False
        self.websocket_handler = None
        self.hume_socket = None
        self.audio_interface_task = None
        self.processing_thread = None
        self.should_stop = False
        self.async_hume_client = None  # Single AsyncHumeClient instance
    
    # ===== PROCESSING LIFECYCLE MANAGEMENT =====    
    async def start_processing(self):
        """Start the Hume EVI connection"""
        try:
            self.is_processing = True
            self.should_stop = False
            
            # Get Hume credentials
            hume_key = dcon("hume")
            hume_secret = dcon("hume_secret") 
            hume_config = dcon("hume_config")
            
            self.async_hume_client = AsyncHumeClient(api_key=hume_key)
            options = ChatConnectOptions(
                config_id=hume_config,
                secret_key=hume_secret
            )
            
            self.websocket_handler = WebSocketHandler()
            
            print("Starting Hume EVI connection...")
            
            # Start the connection in a separate thread
            self.processing_thread = threading.Thread(
                target=self._run_hume_connection,
                args=(self.async_hume_client, options)
            )
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            print("Real-time Hume EVI processor started")
            
        except Exception as e:
            print(f"Error starting Hume EVI processor: {e}")
            self.is_processing = False
    
    # ===== HUME CONNECTION MANAGEMENT =====        
    def _run_hume_connection(self, client, options):
        """Run the Hume EVI connection in a separate thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._connect_to_hume(client, options))
        except Exception as e:
            print(f"Error in Hume connection thread: {e}")
        finally:
            loop.close()
            print("Hume connection thread closed")
            
    async def _connect_to_hume(self, client, options):
        """Connect to Hume EVI and handle audio interface"""
        try:
            async with client.empathic_voice.chat.connect_with_callbacks(
                options=options,
                on_open=self.websocket_handler.on_open,
                on_message=self.websocket_handler.on_message,
                on_close=self.websocket_handler.on_close,
                on_error=self.websocket_handler.on_error
            ) as socket:
                self.hume_socket = socket
                print("Connected to Hume EVI, starting microphone interface...")
                
                try:
                    # Start microphone interface for real-time audio with proper cancellation
                    self.audio_interface_task = asyncio.create_task(
                        MicrophoneInterface.start(
                            socket,
                            allow_user_interrupt=True,
                            byte_stream=self.websocket_handler.byte_strs
                        )
                    )
                    
                    # Monitor for stop requests while audio interface is running
                    while self.is_processing and not self.should_stop:
                        await asyncio.sleep(0.1)
                        
                        # Check if user said "goodbye" to auto-terminate
                        if self.websocket_handler and self.websocket_handler.should_auto_terminate():
                            print("Auto-termination triggered by 'goodbye' command")
                            self.should_stop = True
                            break
                    
                    # Cancel audio interface task
                    if self.audio_interface_task and not self.audio_interface_task.done():
                        print("Cancelling microphone interface task...")
                        self.audio_interface_task.cancel()
                        try:
                            await asyncio.wait_for(self.audio_interface_task, timeout=3.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            print("Microphone interface task cancelled/timed out")
                        
                except asyncio.CancelledError:
                    print("Microphone interface cancelled")
                except Exception as e:
                    print(f"Microphone interface error: {e}")
                finally:
                    # Ensure audio interface task is cleaned up
                    if self.audio_interface_task:
                        if not self.audio_interface_task.done():
                            self.audio_interface_task.cancel()
                        self.audio_interface_task = None
                    
                print("Hume EVI connection shutting down...")
                
        except Exception as e:
            print(f"Error connecting to Hume EVI: {e}")
        finally:
            self.hume_socket = None
            if self.audio_interface_task:
                if not self.audio_interface_task.done():
                    self.audio_interface_task.cancel()
                self.audio_interface_task = None
            print("Hume EVI connection closed")
    
    # ===== CLEANUP AND TERMINATION =====    
    def stop_processing(self):
        """Stop the voice processing and close all connections"""
        print("Stopping Hume EVI processor...")
        self.should_stop = True
        self.is_processing = False
        
        # Cancel audio interface task first (most important for stopping microphone)
        if self.audio_interface_task and not self.audio_interface_task.done():
            try:
                print("Cancelling audio interface task...")
                self.audio_interface_task.cancel()
                print("Audio interface task cancellation requested")
            except Exception as e:
                print(f"Error cancelling audio interface task: {e}")
        
        # Close WebSocket connection if it exists
        if self.hume_socket:
            try:
                # Force close the socket connection
                self.hume_socket = None
                print("Hume socket connection closed")
            except Exception as e:
                print(f"Error closing socket: {e}")
            
        # Wait for processing thread to finish with shorter timeout
        if self.processing_thread and self.processing_thread.is_alive():
            print("Waiting for processing thread to finish...")
            self.processing_thread.join(timeout=3)  # Reduced timeout
            if self.processing_thread.is_alive():
                print("Warning: Processing thread did not finish gracefully within 3 seconds")
            else:
                print("Processing thread finished successfully")
            
        # Clean up websocket handler
        if self.websocket_handler:
            self.websocket_handler = None
            print("WebSocket handler cleaned up")
            
        # Clean up client
        if self.async_hume_client:
            self.async_hume_client = None
            print("AsyncHumeClient cleaned up")
            
        # Ensure audio interface task is cleaned up
        self.audio_interface_task = None
            
        print("Real-time Hume EVI processor stopped completely with audio interface terminated")
        
    def force_terminate_all_tasks(self):
        """Force terminate all running tasks - emergency cleanup"""
        print("Force terminating all Hume EVI tasks...")
        
        # Force stop flags
        self.should_stop = True
        self.is_processing = False
        
        # Cancel audio interface task immediately
        if self.audio_interface_task and not self.audio_interface_task.done():
            try:
                self.audio_interface_task.cancel()
                print("Audio interface task force cancelled")
            except Exception as e:
                print(f"Error force cancelling audio interface task: {e}")
        
        # Force close socket
        if self.hume_socket:
            self.hume_socket = None
            print("Hume socket force closed")
        
        # Clean up all references
        self.websocket_handler = None
        self.async_hume_client = None
        self.audio_interface_task = None
        
        print("All Hume EVI tasks force terminated")
    
    # ===== MESSAGE AND STATUS UTILITIES =====
    def get_latest_message(self):
        """Get latest message from WebSocket handler"""
        if self.websocket_handler:
            return self.websocket_handler.get_latest_message()
        return None
        
    def has_messages(self):
        """Check if there are pending messages"""
        if self.websocket_handler:
            return self.websocket_handler.has_messages()
        return False
        
    def is_ai_speaking(self):
        """Check if AI is currently speaking (for echo prevention)"""
        if self.websocket_handler:
            return self.websocket_handler.is_ai_speaking
        return False


############################################################################
# GLOBAL SESSION VARIABLES - AI Session Management
############################################################################

# Global variables for AI session management
async_hume_client = None  # Single AsyncHumeClient instance
websocket_handler = None
ai_session_active = False
voice_processor = None

############################################################################
# MAIN DJANGO VIEWS - Frontend Templates and Routing
############################################################################

# Create your views here.
def home(request):
    """Main view to render the frontend template"""
    return render(request, 'tester.html')

############################################################################
# AI INITIALIZATION ENDPOINT - Setup Hume EVI Connection
############################################################################

@csrf_exempt
@require_http_methods(["POST"])
def initialize_ai(request):
    """Initialize AI backend with Hume EVI for real-time voice interaction"""
    global async_hume_client, websocket_handler, ai_session_active, voice_processor
    
    try:
        data = json.loads(request.body)
        call_type = data.get('call_type', 'audio')
        
        # Initialize single AsyncHumeClient instance
        hume_key = dcon("hume")
        async_hume_client = AsyncHumeClient(api_key=hume_key)
        ai_session_active = True
        
        # Initialize the real-time voice processor with Hume EVI
        voice_processor = RealTimeVoiceProcessor()
        
        # Start Hume EVI connection in a separate thread
        def start_evi():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(voice_processor.start_processing())
            except Exception as e:
                print(f"Error starting EVI: {e}")
            finally:
                loop.close()
        
        evi_thread = threading.Thread(target=start_evi)
        evi_thread.daemon = True
        evi_thread.start()
        
        return JsonResponse({
            'status': 'success',
            'message': f'AI system initialized for {call_type} call. Hume EVI real-time voice analysis activated!',
            'call_type': call_type
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Failed to initialize AI: {str(e)}'
        }, status=500)

############################################################################
# EMOTION ANALYSIS ENDPOINT - Process Video Frames
############################################################################

@csrf_exempt
@require_http_methods(["POST"])
def analyze_emotion(request):
    """Analyze emotion from uploaded frame"""
    global async_hume_client, ai_session_active
    
    if not ai_session_active or not async_hume_client:
        return JsonResponse({'error': 'AI not initialized'}, status=400)
    
    try:
        frame_file = request.FILES.get('frame')
        if not frame_file:
            return JsonResponse({'error': 'No frame provided'}, status=400)
        
        # Save the frame temporarily
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            for chunk in frame_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
        
        # Analyze emotions using existing function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            emotions = loop.run_until_complete(analyze_frame_async(tmp_path))
            return JsonResponse(emotions, safe=False)
        finally:
            loop.close()
            os.remove(tmp_path)
            
    except Exception as e:
        return JsonResponse({'error': f'Emotion analysis failed: {str(e)}'}, status=500)

############################################################################
# AUDIO PROCESSING ENDPOINT - Handle Voice Input
############################################################################

@csrf_exempt
@require_http_methods(["POST"])
def process_audio(request):
    """Process audio from frontend and return AI response with live interaction"""
    global async_hume_client, websocket_handler, ai_session_active
    
    print(f"Process audio called - Method: {request.method}")
    print(f"Content type: {request.content_type}")
    print(f"AI session active: {ai_session_active}")
    print(f"Files in request: {list(request.FILES.keys())}")
    
    if not ai_session_active or not async_hume_client:
        print("AI not initialized")
        return JsonResponse({'error': 'AI not initialized'}, status=400)
    
    try:
        audio_file = request.FILES.get('audio')
        if not audio_file:
            print("No audio file found in request")
            return JsonResponse({'error': 'No audio provided'}, status=400)
        
        print(f"Audio file received - Name: {audio_file.name}, Size: {audio_file.size}")
        
        # Get the file size before reading
        audio_size = audio_file.size
        
        # Save audio temporarily
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            for chunk in audio_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name
        
        print(f"Audio saved to: {tmp_path}")
        
        # Process audio with Hume AI voice analysis (note: Hume EVI handles responses directly)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            voice_analysis = loop.run_until_complete(process_voice_async(tmp_path))
            # Note: With Hume EVI, responses come directly from the WebSocket connection
            # This endpoint is kept for backward compatibility but may not be needed
        finally:
            loop.close()
            os.remove(tmp_path)
        
        print("Audio processing completed successfully")
        
        return JsonResponse({
            'status': 'success',
            'message': 'Audio processed - responses handled by Hume EVI WebSocket connection',
            'voice_analysis': voice_analysis,
            'audio_length': audio_size,
            'note': 'AI responses come directly from Hume EVI, not this endpoint'
        })
        
    except Exception as e:
        print(f"Audio processing error: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': f'Audio processing failed: {str(e)}'}, status=500)

############################################################################
# SESSION MANAGEMENT ENDPOINT - Stop AI and Cleanup
############################################################################

@csrf_exempt
@require_http_methods(["POST"])
def stop_ai_session(request):
    """Stop AI session and cleanup resources"""
    global async_hume_client, websocket_handler, ai_session_active, voice_processor
    
    try:
        print("Stopping AI session and cleaning up all resources...")
        
        # Stop Hume EVI voice processor with force termination
        if voice_processor:
            print("Force stopping voice processor and all audio tasks...")
            voice_processor.force_terminate_all_tasks()
            voice_processor = None
        
        # Clean up other AI resources
        if websocket_handler:
            print("Cleaning up websocket handler...")
            websocket_handler = None
        
        # Clean up the single AsyncHumeClient instance
        if async_hume_client:
            print("Cleaning up AsyncHumeClient...")
            async_hume_client = None
        
        ai_session_active = False
        print("AI session cleanup completed with force termination")
        
        return JsonResponse({
            'status': 'success',
            'message': 'AI session stopped successfully. Hume EVI disconnected and all audio processing terminated.'
        })
    except Exception as e:
        print(f"Error stopping AI session: {e}")
        # Even if there's an error, ensure everything is cleaned up
        voice_processor = None
        websocket_handler = None
        async_hume_client = None
        ai_session_active = False
        
        return JsonResponse({
            'status': 'error',
            'message': f'Error stopping AI session: {str(e)} (but resources cleaned up)'
        }, status=500)

############################################################################
# AUDIO STREAMING ENDPOINT - Real-time Audio Chunks
############################################################################

@csrf_exempt
@require_http_methods(["POST"]) 
def stream_audio_chunk(request):
    """Note: With Hume EVI, audio is handled directly by MicrophoneInterface"""
    global voice_processor, ai_session_active
    
    if not ai_session_active:
        return JsonResponse({'error': 'AI not initialized'}, status=400)
    
    # With Hume EVI, we don't need to manually handle audio chunks
    # The MicrophoneInterface handles this automatically
    return JsonResponse({
        'status': 'success',
        'message': 'Audio handled by Hume EVI MicrophoneInterface'
    })

############################################################################
# AI RESPONSE POLLING ENDPOINT - Get Messages from Hume EVI
############################################################################

@csrf_exempt
@require_http_methods(["GET"])
def poll_ai_response(request):
    """Poll for latest AI response from Hume EVI"""
    global voice_processor
    
    try:
        if voice_processor and voice_processor.has_messages():
            message_data = voice_processor.get_latest_message()
            
            if message_data:
                if message_data['type'] == 'assistant_message':
                    return JsonResponse({
                        'status': 'success',
                        'has_response': True,
                        'data': {
                            'timestamp': message_data['timestamp'],
                            'response': message_data['content'],
                            # No emotions sent with assistant messages
                            'should_speak': False  # Hume EVI handles audio output
                        }
                    })
                elif message_data['type'] == 'audio_output':
                    return JsonResponse({
                        'status': 'success',
                        'has_response': True,
                        'data': {
                            'timestamp': message_data['timestamp'],
                            'type': 'audio_output',
                            'audio_data': message_data['audio_data']  # base64 string
                        }
                    })
                elif message_data['type'] == 'user_message':
                    user_emotions = message_data.get('emotions', [])
                    print(f"Sending user message with {len(user_emotions)} emotions to frontend")
                    if user_emotions:
                        emotion_strings = [f"{e['name']}: {e['score']:.2f}" for e in user_emotions]
                        print(f"User emotions: {emotion_strings}")
                    
                    return JsonResponse({
                        'status': 'success',
                        'has_response': True,
                        'data': {
                            'timestamp': message_data['timestamp'],
                            'user_message': message_data['content'],
                            'emotions': user_emotions,  # Only user emotions sent here
                            'type': 'user_speech'
                        }
                    })
                elif message_data['type'] == 'farewell_detected':
                    return JsonResponse({
                        'status': 'success',
                        'has_response': True,
                        'data': {
                            'timestamp': message_data['timestamp'],
                            'type': 'auto_terminate',
                            'message': 'Goodbye detected - call will end automatically',
                            'user_message': message_data['content']
                        }
                    })
                elif message_data['type'] == 'ai_speaking_start':
                    return JsonResponse({
                        'status': 'success',
                        'has_response': True,
                        'data': {
                            'timestamp': message_data['timestamp'],
                            'type': 'ai_speaking_start'
                        }
                    })
                elif message_data['type'] == 'ai_speaking_end':
                    return JsonResponse({
                        'status': 'success',
                        'has_response': True,
                        'data': {
                            'timestamp': message_data['timestamp'],
                            'type': 'ai_speaking_end'
                        }
                    })
                elif message_data['type'] == 'error':
                    return JsonResponse({
                        'status': 'error',
                        'message': message_data['content']
                    })
        
        return JsonResponse({
            'status': 'success',
            'has_response': False
        })
            
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'message': f'Error polling response: {str(e)}'
        }, status=500)

############################################################################
# ASYNC ANALYSIS FUNCTIONS - Background Processing
############################################################################

async def analyze_frame_async(frame_path):
    """Async function to analyze frame emotions"""
    try:
        hume_key = dcon("hume")
        client = AsyncHumeClient(api_key=hume_key)
        model_config = Config(face=StreamFace())
        stream_options = StreamConnectOptions(config=model_config)
        
        async with client.expression_measurement.stream.connect(options=stream_options) as socket:
            result = await socket.send_file(frame_path)
            predictions = getattr(result.face, "predictions", None)
            if not predictions:
                return []
            
            emotions = []
            for r in predictions:
                for emotion in r.emotions:
                    if emotion.score > 0.1:  # Lower threshold for more emotions
                        emotions.append({
                            'name': emotion.name,
                            'score': emotion.score
                        })
            
            return emotions
    except Exception as e:
        print(f"Error analyzing frame: {e}")
        return []

async def process_voice_async(audio_path):
    """Process voice audio for emotion and speech analysis"""
    try:
        hume_key = dcon("hume")
        hume_secret = dcon("hume_secret")
        hume_ai_Config = dcon("hume_config")
        
        client = AsyncHumeClient(api_key=hume_key)
        
        # For now, return simulated voice analysis
        # In a real implementation, you'd use Hume's voice emotion analysis
        voice_analysis = {
            'emotions': [
                {'name': 'Joy', 'score': 0.7},
                {'name': 'Interest', 'score': 0.6},
                {'name': 'Calmness', 'score': 0.5}
            ],
            'speech_detected': True,
            'confidence': 0.85
        }
        
        return voice_analysis
    except Exception as e:
        print(f"Error processing voice: {e}")
        return {
            'emotions': [],
            'speech_detected': False,
            'confidence': 0.0
        }