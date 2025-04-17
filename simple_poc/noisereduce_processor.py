#!/usr/bin/env python3
import os
import numpy as np
import torch
import torchaudio
import tempfile
import logging
from pathlib import Path
import noisereduce as nr

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('noisereduce.log')
    ]
)
logger = logging.getLogger('noisereduce')

class NoiseReduceProcessor:
    """
    A processor that uses noisereduce to denoise audio files.
    This implementation uses spectral gating for noise reduction.
    """
    
    def __init__(self, sample_rate: int = 48000, use_torch: bool = False):
        """
        Initialize the NoiseReduce processor.
        
        Args:
            sample_rate: Target sampling rate for audio processing
            use_torch: Whether to use the PyTorch implementation for potential GPU acceleration
        """
        logger.info("Initializing NoiseReduce processor")
        self.sample_rate = sample_rate
        self.use_torch = use_torch
        
        if self.use_torch:
            try:
                from noisereduce.torchgate import TorchGate
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Using TorchGate implementation on {self.device}")
                self.torch_gate = TorchGate(
                    sr=self.sample_rate, 
                    nonstationary=True  # Better for real-time processing with changing noise
                ).to(self.device)
                self.has_torch = True
            except ImportError as e:
                logger.warning(f"PyTorch implementation not available: {e}. Falling back to CPU-based processing.")
                self.has_torch = False
        else:
            logger.info("Using CPU-based noisereduce implementation")
            self.has_torch = False
    
    def process_audio(self, audio_path: str, output_path: str = None) -> str:
        """
        Process an audio file to reduce noise using NoiseReduce.
        
        Args:
            audio_path: Path to the input audio file
            output_path: Path to save the processed audio (WAV)
            
        Returns:
            Path to the processed denoised audio file
        """
        logger.info(f"Processing audio file with NoiseReduce: {audio_path}")
        
        # Use absolute paths to avoid working directory issues
        audio_path = os.path.abspath(audio_path)
        
        # Verify input file exists
        if not os.path.exists(audio_path):
            error_msg = f"Input audio file not found: {audio_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Set output path if not provided
        if output_path is None:
            output_path = os.path.join(os.path.dirname(audio_path), 
                                      f"{Path(audio_path).stem}_denoised.wav")
        else:
            output_path = os.path.abspath(output_path)
        
        try:
            # Process with different methods based on configuration
            if self.has_torch and self.use_torch:
                logger.info("Using TorchGate implementation")
                
                # Load audio file
                waveform, orig_sr = torchaudio.load(audio_path)
                
                # Resample if needed
                if orig_sr != self.sample_rate:
                    logger.info(f"Resampling from {orig_sr}Hz to {self.sample_rate}Hz")
                    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=self.sample_rate)
                    waveform = resampler(waveform)
                
                # Move to GPU if available
                waveform = waveform.to(self.device)
                
                # Process with TorchGate
                denoised_waveform = self.torch_gate(waveform)
                
                # Move back to CPU for saving
                denoised_waveform = denoised_waveform.cpu()
                
                # Save to output file
                torchaudio.save(output_path, denoised_waveform, self.sample_rate)
                
            else:
                logger.info("Using CPU-based noisereduce implementation")
                
                # Load audio data using torchaudio
                waveform, orig_sr = torchaudio.load(audio_path)
                
                # Convert to mono if it's stereo
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                # Convert to numpy
                audio_np = waveform.squeeze().numpy()
                
                # Process with noisereduce (non-stationary mode is better for real-time processing)
                denoised_audio = nr.reduce_noise(
                    y=audio_np,
                    sr=orig_sr,
                    stationary=False,
                    time_constant_s=1.0,  # Balance between noise detection and real-time processing
                    n_fft=2048,          # Larger FFT for music, 512 for speech
                    n_jobs=-1,           # Use all CPU cores
                    chunk_size=60000,    # Adjust based on memory constraints
                    padding=2000         # Padding for overlapping chunks
                )
                
                # Convert back to torch tensor
                denoised_tensor = torch.from_numpy(denoised_audio).unsqueeze(0)
                
                # Save to output file
                torchaudio.save(output_path, denoised_tensor, orig_sr)
            
            logger.info(f"Denoising completed, output saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error processing audio with NoiseReduce: {e}", exc_info=True)
            raise
    
    def process_audio_tensor(self, audio_tensor: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Process an audio tensor to reduce noise using NoiseReduce.
        For real-time streaming applications.
        
        Args:
            audio_tensor: Audio tensor (channels, samples)
            sr: Sample rate of the audio tensor
            
        Returns:
            Denoised audio tensor
        """
        logger.info(f"Processing audio tensor with shape {audio_tensor.shape}")
        
        try:
            if self.has_torch and self.use_torch:
                # Make sure tensor is on the correct device
                audio_tensor = audio_tensor.to(self.device)
                
                # Process with TorchGate
                denoised_tensor = self.torch_gate(audio_tensor)
                
                # Return to original device (likely CPU)
                return denoised_tensor.cpu()
            else:
                # Convert to numpy for processing
                is_1d = (audio_tensor.dim() == 1)
                if is_1d:
                    audio_tensor = audio_tensor.unsqueeze(0)
                
                # Convert to mono if stereo
                if audio_tensor.shape[0] > 1:
                    audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
                
                # Convert to numpy
                audio_np = audio_tensor.squeeze().numpy()
                
                # Process with noisereduce
                denoised_audio = nr.reduce_noise(
                    y=audio_np,
                    sr=sr,
                    stationary=False,
                    time_constant_s=0.5,  # Shorter time constant for real-time processing
                    n_fft=1024,
                    n_jobs=1             # Single job for real-time processing
                )
                
                # Convert back to tensor with correct dimensions
                if is_1d:
                    return torch.from_numpy(denoised_audio)
                else:
                    return torch.from_numpy(denoised_audio).unsqueeze(0)
                
        except Exception as e:
            logger.error(f"Error processing audio tensor: {e}", exc_info=True)
            # In case of error, return the original tensor
            return audio_tensor
