#!/usr/bin/env python3
import os
import time
import logging
import subprocess
from simple_poc.silero_vad_processor import SileroVADProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('process_audio.log')
    ]
)
logger = logging.getLogger('process_audio')

def main():
    # Hardcoded input file (MP3)
    input_file = "weather.mp3"
    output_file = "weather_voice_only.wav"
    
    # VAD parameters
    threshold = 0.5
    min_speech_ms = 250
    min_silence_ms = 500
    
    # Check if input file exists
    if not os.path.exists(input_file):
        logger.error(f"Error: Input file '{input_file}' does not exist")
        return 1
    
    logger.info(f"Processing file: {input_file}")
    start_time = time.time()
    
    # Initialize processor
    processor = SileroVADProcessor(
        threshold=threshold,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms
    )
    
    try:
        # Process audio first
        output_path = processor.process_audio(input_file, output_file)
        logger.info(f"Successfully processed audio and saved to: {output_path}")
        
        # Now get speech segments for reporting
        try:
            # Get speech segments and print stats
            segments = processor.get_speech_segments(input_file)
            logger.info(f"Found {len(segments)} speech segments:")
            
            total_speech = 0
            for i, (start, end) in enumerate(segments, 1):
                duration = end - start
                total_speech += duration
                logger.info(f"  {i}. {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")
            
            # Calculate audio duration (approximately)
            try:
                cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                      '-of', 'default=noprint_wrappers=1:nokey=1', input_file]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                input_duration = float(result.stdout.strip())
                speech_percentage = (total_speech / input_duration) * 100
                logger.info(f"Total speech: {total_speech:.2f}s out of {input_duration:.2f}s ({speech_percentage:.1f}%)")
            except Exception as e:
                logger.warning(f"Could not determine exact audio duration: {e}")
                logger.info(f"Total speech: {total_speech:.2f}s")
        except Exception as e:
            logger.error(f"Error getting speech segments: {e}")

        # Processing time stats
        elapsed_time = time.time() - start_time
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        
        return 0
    except Exception as e:
        logger.error(f"Error processing audio: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main()) 