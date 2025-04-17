import uvicorn
import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check for required environment variables
if 'OPENAI_API_KEY' not in os.environ:
    print("WARNING: OPENAI_API_KEY environment variable not set. Transcription will fail.")
    print("Please create a .env file in the project root with your OpenAI API key.")
    print("Example: OPENAI_API_KEY=your-api-key-here")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    print("Starting Audio Transcription WebSocket Server...")
    print("Connect to ws://localhost:8000/ws/{client_id} to stream audio")
    
    # Create transcriptions directory if it doesn't exist
    os.makedirs("./transcriptions", exist_ok=True)
    
    # Run the server using uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True) 