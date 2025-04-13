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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('silero_vad.log')
    ]
)
logger = logging.getLogger('silero_vad')

class SileroVADProcessor:
    """
    A processor that uses Silero VAD to detect and extract human voice segments from audio files.
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
        
    def convert_audio_to_wav(self, audio_path: str) -> str:
        """
        Convert an audio file (MP3/MP4) to WAV format using ffmpeg.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Path to the converted WAV file
        """
        file_ext = Path(audio_path).suffix.lower()
        logger.info(f"Converting {file_ext} file to WAV: {audio_path}")
        
        # Create temporary file for WAV output
        temp_dir = tempfile.gettempdir()
        wav_path = os.path.join(temp_dir, f"{Path(audio_path).stem}.wav")
        
        # Convert using ffmpeg
        cmd = [
            'ffmpeg', '-i', audio_path, 
            '-ar', str(self.sampling_rate), 
            '-ac', '1',  # Mono channel
            '-y',  # Overwrite output file if it exists
            wav_path
        ]
        
        logger.info(f"Running ffmpeg command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info(f"Conversion successful: {wav_path}")
        return wav_path
    
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
    
    def process_audio(self, audio_path: str, output_path: str = None) -> str:
        """
        Process an audio file to extract only human voice segments.
        
        Args:
            audio_path: Path to the input audio file (MP3/MP4)
            output_path: Path to save the processed audio (WAV)
            
        Returns:
            Path to the processed audio file
        """
        logger.info(f"Processing audio file: {audio_path}")
        
        # Convert audio to WAV if needed
        file_ext = Path(audio_path).suffix.lower()
        if file_ext in ['.mp3', '.mp4', '.m4a', '.aac']:
            wav_path = self.convert_audio_to_wav(audio_path)
        else:
            wav_path = audio_path
            
        # Load audio
        logger.info(f"Loading audio from {wav_path}")
        wav, sr = torchaudio.load(wav_path)
        
        # Convert to mono if necessary
        if wav.shape[0] > 1:
            logger.info("Converting stereo to mono")
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        # Resample if needed
        if sr != self.sampling_rate:
            logger.info(f"Resampling audio from {sr}Hz to {self.sampling_rate}Hz")
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
            wav = resampler(wav)
            sr = self.sampling_rate
            
        # Get speech timestamps
        logger.info("Detecting speech segments")
        speech_timestamps = self.get_speech_timestamps(
            wav[0],
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sampling_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms
        )
        
        logger.info(f"Found {len(speech_timestamps)} speech segments")
        
        # Set output path if not provided
        if output_path is None:
            output_path = os.path.join(os.path.dirname(audio_path), 
                                      f"{Path(audio_path).stem}_voice_only.wav")
        
        # Create audio with only speech segments
        logger.info("Collecting speech chunks")
        
        if len(speech_timestamps) > 0:
            # Manually collect speech chunks from the original audio
            logger.info("Manually collecting speech chunks")
            wav_chunks = self.collect_speech_chunks(speech_timestamps, wav[0])
            
            # Save the collected chunks
            logger.info(f"Saving processed audio to {output_path}")
            torchaudio.save(
                output_path,
                wav_chunks.unsqueeze(0),
                sample_rate=self.sampling_rate
            )
        else:
            logger.warning("No speech detected in the audio file")
            # Save an empty audio file
            empty_wav = torch.zeros(1, 1000)  # 1000 samples of silence
            torchaudio.save(output_path, empty_wav, sample_rate=self.sampling_rate)
        
        # Clean up temporary file if we created one
        if file_ext in ['.mp3', '.mp4', '.m4a', '.aac'] and os.path.exists(wav_path):
            logger.info(f"Removing temporary file: {wav_path}")
            os.remove(wav_path)
            
        logger.info(f"Processing completed. Output file: {output_path}")
        return output_path
    
    def get_speech_segments(self, audio_path: str) -> List[Tuple[float, float]]:
        """
        Get time segments (in seconds) that contain speech.
        
        Args:
            audio_path: Path to the input audio file
            
        Returns:
            List of tuples with (start_time, end_time) in seconds
        """
        logger.info(f"Getting speech segments from: {audio_path}")
        
        # Convert audio to WAV if needed
        file_ext = Path(audio_path).suffix.lower()
        if file_ext in ['.mp3', '.mp4', '.m4a', '.aac']:
            wav_path = self.convert_audio_to_wav(audio_path)
        else:
            wav_path = audio_path
            
        # Load audio
        wav, sr = torchaudio.load(wav_path)
        
        # Convert to mono if necessary
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        # Resample if needed
        if sr != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
            wav = resampler(wav)
            sr = self.sampling_rate
            
        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            wav[0],
            self.model,
            threshold=self.threshold,
            sampling_rate=self.sampling_rate,
            min_speech_duration_ms=self.min_speech_duration_ms,
            min_silence_duration_ms=self.min_silence_duration_ms
        )
        
        # Convert frame indices to time in seconds
        segments = []
        for segment in speech_timestamps:
            start_time = segment['start'] / self.sampling_rate
            end_time = segment['end'] / self.sampling_rate
            segments.append((start_time, end_time))
        
        logger.info(f"Found {len(segments)} speech segments")
            
        # Clean up temporary file if we created one
        if file_ext in ['.mp3', '.mp4', '.m4a', '.aac'] and os.path.exists(wav_path):
            logger.info(f"Removing temporary file: {wav_path}")
            os.remove(wav_path)
            
        return segments 