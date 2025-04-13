#!/usr/bin/env python3
import os
import time
import logging
from noisereduce_processor import NoiseReduceProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('simple_test_noisereduce.log')
    ]
)
logger = logging.getLogger('simple_test_noisereduce')

def main():
    # Fixed input and output files
    input_file = "weather_voice_only.wav"
    output_file = "weather_denoised.wav"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Error: Input file '{input_file}' does not exist")
        return 1
    
    logger.info(f"Processing file: {input_file}")
    logger.info("This will take a few moments depending on the file size...")
    start_time = time.time()
    
    # Initialize processor (using CPU-based processing by default)
    logger.info("Initializing NoiseReduce processor...")
    processor = NoiseReduceProcessor(
        sample_rate=48000,  # 48kHz is standard for good quality audio
        use_torch=False     # Use CPU-based processing for simplicity
    )
    
    try:
        # Process audio with NoiseReduce
        logger.info("Starting denoising process...")
        output_path = processor.process_audio(input_file, output_file)
        
        # Calculate file sizes for reporting
        input_size = os.path.getsize(input_file) / (1024 * 1024)  # MB
        output_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        
        # Processing time stats
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Input file size: {input_size:.2f} MB")
        logger.info(f"Output file size: {output_size:.2f} MB")
        logger.info(f"Output saved to: {output_path}")
        
        print("\n=========== DENOISING COMPLETE ===========")
        print(f"Input file: {input_file} ({input_size:.1f} MB)")
        print(f"Output file: {output_path} ({output_size:.1f} MB)")
        print(f"Processing time: {elapsed_time:.1f} seconds")
        print("==========================================\n")
        print(f"Listen to {output_path} to hear the denoised result!")
        
        return 0
    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        return 1

if __name__ == "__main__":
    print("\n======= NOISEREDUCE AUDIO PROCESSOR =======")
    print("Starting simple denoising test...")
    exit(main())
