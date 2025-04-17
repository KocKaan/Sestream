import torch
import torchaudio
import numpy as np
import os
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import List, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('vad_processor')

class SileroVADProcessor:
    """
    A processor that uses Silero VAD to detect and extract human voice segments from audio data.
    """
    
    def __init__(self, threshold: float = 0.5, min_speech_duration_ms: int = 250, 
                 min_silence_duration_ms: int = 500, sampling_rate: int = 16000):
        """
        Initialize the Silero VAD processor.
        
        Args:
            threshold: Confidence threshold for VAD (0 to 1)
            min_speech_duration_ms: Minimum duration of speech segments to keep
            min_silence_duration_ms: Minimum duration of silence between speech segments
            sampling_rate: Target sampling rate for audio processing
        """
        logger.info("Initializing Silero VAD processor")
        # Download and initialize Silero VAD model
        logger.info("Loading Silero VAD model from torch hub")
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        
        self.threshold = threshold
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.sampling_rate = sampling_rate
        
        # Get the VAD functions from utils
        self.get_speech_timestamps = utils[0]
        
        logger.info(f"Silero VAD initialized with threshold={threshold}, min_speech={min_speech_duration_ms}ms, min_silence={min_silence_duration_ms}ms")
    
    def collect_speech_chunks(self, speech_timestamps, audio_tensor):
        """
        Manually collect speech chunks from audio tensor using timestamps.
        
        Args:
            speech_timestamps: List of speech segment dictionaries with 'start' and 'end' keys
            audio_tensor: Tensor containing audio data
            
        Returns:
            Tensor with concatenated speech segments
        """
        chunks = []
        for ts in speech_timestamps:
            start_frame = ts['start']
            end_frame = ts['end']
            chunks.append(audio_tensor[start_frame:end_frame])
        
        if not chunks:
            return torch.zeros(0)
        
        # Concatenate all chunks along time dimension
        return torch.cat(chunks, dim=0)
    
    def process_audio_tensor(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process an audio tensor to extract only human voice segments.
        
        Args:
            audio_tensor: Tensor containing audio data (should be at self.sampling_rate)
            
        Returns:
            Tensor containing only the voice segments
        """
        logger.info(f"Processing audio tensor with shape {audio_tensor.shape}")
        
        # Make sure tensor is 1D (mono)
        if audio_tensor.dim() > 1:
            if audio_tensor.shape[0] > 1:  # If it's a batch or has channels
                audio_tensor = torch.mean(audio_tensor, dim=0)
            else:
                audio_tensor = audio_tensor.squeeze(0)
        
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sampling_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms
        )
        
        logger.info(f"Found {len(speech_timestamps)} speech segments")
        
        # Create audio with only speech segments
        if len(speech_timestamps) > 0:
            # Manually collect speech chunks from the original audio
            voice_only_tensor = self.collect_speech_chunks(speech_timestamps, audio_tensor)
            return voice_only_tensor
        else:
            logger.warning("No speech detected in the audio data")
            return torch.zeros(0) 