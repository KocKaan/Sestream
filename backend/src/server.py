import os
import asyncio
import logging
import json
import torch
import numpy as np
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import base64
from io import BytesIO
from datetime import datetime
from .vad_processor import SileroVADProcessor
from .noise_processor import NoiseReduceProcessor
from .transcription import OpenAITranscriber

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('server')

# Create FastAPI app
app = FastAPI(
    title="Audio Transcription Server",
    description="WebSocket server for real-time audio processing and transcription",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Debug mode flag
DEBUG_MODE = True
DEBUG_DIR = "./debug_audio"
os.makedirs(DEBUG_DIR, exist_ok=True)

# Complete audio directory
COMPLETE_AUDIO_DIR = os.path.join(DEBUG_DIR, "complete_audio")
os.makedirs(COMPLETE_AUDIO_DIR, exist_ok=True)

# Audio buffering settings
BUFFER_MAX_SECONDS = 5.0  # Maximum seconds to buffer before processing
BUFFER_MIN_SECONDS = 1.5  # Minimum seconds required before processing

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.sample_rate = 16000
        
        # Audio buffers for each client
        self.audio_buffers: Dict[str, List[torch.Tensor]] = {}
        self.buffer_sizes: Dict[str, int] = {}  # Track total samples in buffer for each client

        # Complete audio recordings for debug
        self.complete_recordings: Dict[str, List[torch.Tensor]] = {}  # All raw audio from client
        self.transcription_audio: Dict[str, List[torch.Tensor]] = {}  # All audio sent to transcription

        # Silence detection threshold (RMS)
        self.silence_threshold = 0.01
        
        # Initialize the processing components
        self.vad_processor = SileroVADProcessor(
            threshold=0.3,  # More sensitive to detect speech
            min_speech_duration_ms=250,
            min_silence_duration_ms=1000,
            sampling_rate=self.sample_rate
        )
        
        self.noise_processor = NoiseReduceProcessor(
            sample_rate=self.sample_rate,
            use_torch=False,
            prop_decrease=0.75,  # Balanced noise reduction to preserve speech
            n_fft=1024
        )
        
        self.transcriber = OpenAITranscriber(
            sample_rate=self.sample_rate,
            transcription_dir="./transcriptions"
        )
        
        logger.info("Audio processing pipeline initialized")
        logger.info(f"Server will buffer between {BUFFER_MIN_SECONDS}s and {BUFFER_MAX_SECONDS}s of audio before processing")
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
        # Initialize empty buffer for this client
        self.audio_buffers[client_id] = []
        self.buffer_sizes[client_id] = 0
        
        # Initialize debug recordings
        if DEBUG_MODE:
            self.complete_recordings[client_id] = []
            self.transcription_audio[client_id] = []
        
        logger.info(f"Client {client_id} connected, total connections: {len(self.active_connections)}")
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
            # Save complete recordings if in debug mode
            if DEBUG_MODE:
                self.save_complete_recordings(client_id)
            
            # Clean up buffers for this client
            if client_id in self.audio_buffers:
                del self.audio_buffers[client_id]
            if client_id in self.buffer_sizes:
                del self.buffer_sizes[client_id]
                
            logger.info(f"Client {client_id} disconnected, remaining connections: {len(self.active_connections)}")
    
    def save_complete_recordings(self, client_id: str):
        """Save the complete audio recordings for a client"""
        if not DEBUG_MODE:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete raw audio
        if client_id in self.complete_recordings and self.complete_recordings[client_id]:
            try:
                raw_filename = f"raw_complete_{client_id}_{timestamp}.wav"
                raw_filepath = os.path.join(COMPLETE_AUDIO_DIR, raw_filename)
                
                # Concatenate all audio chunks
                raw_audio = torch.cat(self.complete_recordings[client_id])
                if raw_audio.dim() == 1:
                    raw_audio = raw_audio.unsqueeze(0)
                
                # Normalize if needed
                if raw_audio.abs().max() > 1.0:
                    raw_audio = raw_audio / raw_audio.abs().max()
                
                # Save to file
                torchaudio.save(raw_filepath, raw_audio, self.sample_rate)
                logger.info(f"Saved complete raw audio: {raw_filepath}")
                
                # Clean up
                del self.complete_recordings[client_id]
                
            except Exception as e:
                logger.error(f"Error saving complete raw audio: {e}")
                
        # Save complete transcription audio
        if client_id in self.transcription_audio and self.transcription_audio[client_id]:
            try:
                trans_filename = f"transcription_complete_{client_id}_{timestamp}.wav"
                trans_filepath = os.path.join(COMPLETE_AUDIO_DIR, trans_filename)
                
                # Concatenate all audio chunks
                trans_audio = torch.cat(self.transcription_audio[client_id])
                if trans_audio.dim() == 1:
                    trans_audio = trans_audio.unsqueeze(0)
                
                # Normalize if needed
                if trans_audio.abs().max() > 1.0:
                    trans_audio = trans_audio / trans_audio.abs().max()
                
                # Save to file
                torchaudio.save(trans_filepath, trans_audio, self.sample_rate)
                logger.info(f"Saved complete transcription audio: {trans_filepath}")
                
                # Clean up
                del self.transcription_audio[client_id]
                
            except Exception as e:
                logger.error(f"Error saving complete transcription audio: {e}")
            
    async def send_message(self, client_id: str, message: Dict[str, Any]):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)
    
    def save_debug_audio(self, audio_tensor, step_name, client_id):
        """Save audio tensor to WAV file for debugging"""
        if not DEBUG_MODE:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{step_name}_{client_id}_{timestamp}.wav"
        filepath = os.path.join(DEBUG_DIR, filename)
        
        # Ensure tensor has correct shape for saving
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Normalize if needed
        if audio_tensor.abs().max() > 1.0:
            audio_tensor = audio_tensor / audio_tensor.abs().max()
            
        try:
            torchaudio.save(filepath, audio_tensor, self.sample_rate)
            logger.info(f"Saved debug audio: {filepath}")
        except Exception as e:
            logger.error(f"Error saving debug audio: {e}")
            
    def get_buffer_duration(self, client_id):
        """Get the current buffer duration in seconds"""
        if client_id in self.buffer_sizes:
            return self.buffer_sizes[client_id] / self.sample_rate
        return 0
    
    def add_to_buffer(self, client_id, audio_tensor):
        """Add audio tensor to the buffer for the given client"""
        if client_id not in self.audio_buffers:
            self.audio_buffers[client_id] = []
            self.buffer_sizes[client_id] = 0
            
        # Add to buffer
        self.audio_buffers[client_id].append(audio_tensor)
        self.buffer_sizes[client_id] += audio_tensor.numel()
        
        # Add to complete recording if in debug mode
        if DEBUG_MODE and client_id in self.complete_recordings:
            self.complete_recordings[client_id].append(audio_tensor.clone())
        
        buffer_duration = self.get_buffer_duration(client_id)
        logger.info(f"Added {audio_tensor.numel()} samples to buffer. Buffer now has {buffer_duration:.2f}s of audio")
    
    def is_buffer_ready(self, client_id):
        """Check if buffer has enough audio to process"""
        buffer_duration = self.get_buffer_duration(client_id)
        return buffer_duration >= BUFFER_MIN_SECONDS
    
    def should_process_buffer(self, client_id, audio_tensor):
        """Determine if we should process the buffer now based on buffer size and silence detection"""
        buffer_duration = self.get_buffer_duration(client_id)
        
        # Process if buffer exceeds max duration
        if buffer_duration >= BUFFER_MAX_SECONDS:
            logger.info(f"Buffer reached maximum size ({buffer_duration:.2f}s), processing now")
            return True
            
        # Process if buffer has minimum data and current chunk ends with silence
        if buffer_duration >= BUFFER_MIN_SECONDS:
            # Calculate RMS of the last ~200ms to detect silence
            samples_to_check = min(int(0.2 * self.sample_rate), audio_tensor.numel())
            if samples_to_check > 0:
                end_segment = audio_tensor[-samples_to_check:]
                rms = torch.sqrt(torch.mean(end_segment**2))
                
                if rms < self.silence_threshold:
                    logger.info(f"Detected silence at end of audio chunk (RMS: {rms:.4f}), processing buffer ({buffer_duration:.2f}s)")
                    return True
        
        return False
    
    def get_buffer_content(self, client_id):
        """Get concatenated buffer content and clear the buffer"""
        if client_id not in self.audio_buffers or not self.audio_buffers[client_id]:
            return torch.zeros(0)
            
        # Concatenate all tensors in the buffer
        buffer_tensors = self.audio_buffers[client_id]
        concatenated = torch.cat(buffer_tensors)
        
        # Clear the buffer
        self.audio_buffers[client_id] = []
        self.buffer_sizes[client_id] = 0
        
        return concatenated
    
    async def process_audio_chunk(self, audio_data: bytes, client_id: str):
        """
        Process a chunk of audio data from the WebSocket client.
        
        Args:
            audio_data: Raw audio data as bytes
            client_id: Identifier for the client connection
        """
        try:
            # Convert the raw bytes to a numpy array (assuming float32 PCM data)
            audio_np = np.frombuffer(audio_data, dtype=np.float32)
            
            # Skip processing if chunk is empty or too small
            if audio_np.size < 100:  # Arbitrary small number
                logger.warning(f"Received very small audio chunk ({audio_np.size} samples), skipping")
                return
                
            # Log info about the incoming audio data
            logger.info(f"Received audio chunk: shape={audio_np.shape}, min={audio_np.min():.4f}, max={audio_np.max():.4f}")
            
            # Convert to PyTorch tensor
            audio_tensor = torch.from_numpy(audio_np)
            
            # Add to the client's buffer
            self.add_to_buffer(client_id, audio_tensor)
            
            # Check if we should process the buffer
            if not self.should_process_buffer(client_id, audio_tensor):
                logger.info("Buffer not ready for processing yet")
                return
                
            # Get and clear the buffer content
            buffered_audio = self.get_buffer_content(client_id)
            buffer_duration = buffered_audio.numel() / self.sample_rate
            
            # Save original buffered audio for debugging
            if DEBUG_MODE:
                self.save_debug_audio(buffered_audio, "buffered", client_id)
                
            logger.info(f"Processing {buffer_duration:.2f}s of buffered audio")
            
            # 1. Voice Activity Detection (VAD)
            voice_only_tensor = self.vad_processor.process_audio_tensor(buffered_audio)
            
            # Save VAD output audio for debugging
            if DEBUG_MODE and voice_only_tensor.numel() > 0:
                self.save_debug_audio(voice_only_tensor, "vad_output", client_id)
            
            # If no voice detected, stop processing
            if voice_only_tensor.numel() == 0:
                logger.info(f"No voice detected in buffered audio from client {client_id}")
                await self.send_message(client_id, {"status": "no_voice_detected"})
                return
            
            # 2. Noise reduction
            denoised_tensor = self.noise_processor.process_audio_tensor(voice_only_tensor)
            
            # Save denoised audio for debugging
            if DEBUG_MODE:
                self.save_debug_audio(denoised_tensor, "denoised", client_id)
                
                # Add denoised audio to transcription recordings for complete log
                if client_id in self.transcription_audio:
                    self.transcription_audio[client_id].append(denoised_tensor.clone())
                
            # Log info about the processing steps
            logger.info(f"Audio processing stats: buffered={buffered_audio.shape}, "
                      f"vad_output={voice_only_tensor.shape}, "
                      f"denoised={denoised_tensor.shape}")
            
            # 3. Transcription
            transcription = self.transcriber.transcribe_audio_tensor(denoised_tensor)
            
            # Send the transcription back to the client
            if transcription:
                logger.info(f"Transcription: '{transcription}'")
                await self.send_message(client_id, {
                    "status": "success",
                    "transcription": transcription
                })
            else:
                await self.send_message(client_id, {"status": "empty_transcription"})
                
        except Exception as e:
            logger.error(f"Error processing audio from client {client_id}: {e}", exc_info=True)
            await self.send_message(client_id, {"status": "error", "message": str(e)})

# Create connection manager
manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from the client
            data = await websocket.receive_bytes()
            
            # Process the audio data asynchronously
            await manager.process_audio_chunk(data, client_id)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Unexpected error in websocket handler for client {client_id}: {e}", exc_info=True)
        manager.disconnect(client_id)

@app.get("/")
async def root():
    return {"message": "Audio Transcription Server is running. Connect to /ws/{client_id} with a WebSocket to start streaming audio."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True) 