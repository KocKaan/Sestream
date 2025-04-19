import os
import numpy as np
import torch
import logging
import noisereduce as nr

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('noise_processor')

class NoiseReduceProcessor:
    """
    A processor that uses noisereduce to denoise audio data.
    """
    
    def __init__(self, sample_rate: int = 16000, use_torch: bool = False, 
                 prop_decrease: float = 0.75, n_fft: int = 1024):
        """
        Initialize the NoiseReduce processor.
        
        Args:
            sample_rate: Target sampling rate for audio processing
            use_torch: Whether to use the PyTorch implementation for potential GPU acceleration
            prop_decrease: Proportion to decrease the noise by (1.0 = remove all noise)
            n_fft: FFT window size. Lower for speech (512-1024), higher for music (2048)
        """
        logger.info("Initializing NoiseReduce processor")
        self.sample_rate = sample_rate
        self.use_torch = use_torch
        self.prop_decrease = prop_decrease
        self.n_fft = n_fft
        
        if self.use_torch:
            try:
                from noisereduce.torchgate import TorchGate
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Using TorchGate implementation on {self.device}")
                self.torch_gate = TorchGate(
                    sr=self.sample_rate, 
                    nonstationary=True,  # Better for real-time processing with changing noise
                    prop_decrease=self.prop_decrease
                ).to(self.device)
                self.has_torch = True
            except ImportError as e:
                logger.warning(f"PyTorch implementation not available: {e}. Falling back to CPU-based processing.")
                self.has_torch = False
        else:
            logger.info(f"Using CPU-based noisereduce implementation with prop_decrease={prop_decrease}, n_fft={n_fft}")
            self.has_torch = False
    
    def process_audio_tensor(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """
        Process an audio tensor to reduce noise using NoiseReduce.
        For real-time streaming applications.
        
        Args:
            audio_tensor: Audio tensor (samples)
            
        Returns:
            Denoised audio tensor
        """
        # Check if audio tensor is empty
        if audio_tensor.numel() == 0:
            logger.warning("Empty audio tensor, skipping denoising")
            return audio_tensor
            
        logger.info(f"Processing audio tensor with shape {audio_tensor.shape}, min={audio_tensor.min().item():.4f}, max={audio_tensor.max().item():.4f}")
        
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
                if not is_1d:
                    if audio_tensor.shape[0] > 1:  # If it has multiple channels
                        audio_tensor = torch.mean(audio_tensor, dim=0)
                    else:
                        audio_tensor = audio_tensor.squeeze(0)
                
                # Convert to numpy
                audio_np = audio_tensor.numpy()
                
                # Process with noisereduce - parameters optimized for speech
                denoised_audio = nr.reduce_noise(
                    y=audio_np,
                    sr=self.sample_rate,
                    stationary=False,  # For changing noise environments
                    prop_decrease=self.prop_decrease,  # How much to reduce noise (0.8-1.0 = aggressive, 0.5-0.7 = balanced)
                    time_constant_s=0.2,  # Shorter time constant for speech (0.2-0.5s)
                    freq_mask_smooth_hz=50,  # Smoothing for frequency mask (50-100 Hz for speech)
                    n_fft=self.n_fft,  # 512-1024 for speech
                    n_jobs=1  # Single job for real-time processing
                )
                
                # Log stats about the denoised audio
                if len(denoised_audio) > 0:
                    logger.info(f"Denoised audio stats: min={np.min(denoised_audio):.4f}, max={np.max(denoised_audio):.4f}")
                
                # Convert back to tensor with correct dimensions
                return torch.from_numpy(denoised_audio.astype(np.float32))
                
        except Exception as e:
            logger.error(f"Error processing audio tensor: {e}", exc_info=True)
            # In case of error, return the original tensor
            return audio_tensor 