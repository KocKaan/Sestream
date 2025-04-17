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

# Connection manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.sample_rate = 16000
        
        # Initialize the processing components
        self.vad_processor = SileroVADProcessor(
            threshold=0.5,
            min_speech_duration_ms=250,
            min_silence_duration_ms=500,
            sampling_rate=self.sample_rate
        )
        
        self.noise_processor = NoiseReduceProcessor(
            sample_rate=self.sample_rate,
            use_torch=False
        )
        
        self.transcriber = OpenAITranscriber(
            sample_rate=self.sample_rate,
            transcription_dir="./transcriptions"
        )
        
        logger.info("Audio processing pipeline initialized")
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected, total connections: {len(self.active_connections)}")
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected, remaining connections: {len(self.active_connections)}")
            
    async def send_message(self, client_id: str, message: Dict[str, Any]):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)
    
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
            
            # Convert to PyTorch tensor
            audio_tensor = torch.from_numpy(audio_np)
            
            # 1. Voice Activity Detection (VAD)
            voice_only_tensor = self.vad_processor.process_audio_tensor(audio_tensor)
            
            # If no voice detected, stop processing
            if voice_only_tensor.numel() == 0:
                logger.info(f"No voice detected in audio chunk from client {client_id}")
                await self.send_message(client_id, {"status": "no_voice_detected"})
                return
            
            # 2. Noise reduction
            denoised_tensor = self.noise_processor.process_audio_tensor(voice_only_tensor)
            
            # 3. Transcription
            transcription = self.transcriber.transcribe_audio_tensor(denoised_tensor)
            
            # Send the transcription back to the client
            if transcription:
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