# Real-time Audio Transcription System

This project implements a real-time audio processing and transcription system using WebSockets. It captures audio from a microphone, processes it with Voice Activity Detection (VAD) and noise reduction, and transcribes it using OpenAI's Whisper model.

## Architecture

The system consists of:

1. A Python WebSocket server (backend) that:
   - Receives audio streams from clients
   - Processes audio with Silero VAD to detect speech
   - Applies noise reduction to clean the audio
   - Transcribes the audio using OpenAI's Whisper API
   - Returns transcriptions to clients

2. A JavaScript client that:
   - Captures audio from the microphone
   - Streams it to the server
   - Displays transcriptions as they are received

## Setup

### Prerequisites

- Python 3.8+
- Node.js (if you want to serve the frontend)
- OpenAI API key (for Whisper transcription)

### Backend Setup

1. Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

2. Create a `.env` file in the project root with your OpenAI API key:

```
OPENAI_API_KEY=your-api-key-here
```

3. Run the server:

```bash
cd backend
python main.py
```

The server will start on `http://localhost:8000`.

### Frontend

The frontend is a simple HTML/JS application. You can open `index.html` directly in a browser, or serve it with a simple HTTP server:

```bash
# Using Python HTTP server
python -m http.server 8080

# Or using Node.js with http-server package
npx http-server -p 8080
```

Then open `http://localhost:8080` in your browser.

## Usage

1. Open the client in your browser
2. Click "Start Recording" to begin capturing audio
3. Speak into your microphone
4. View transcriptions as they appear in real-time
5. Click "Stop Recording" when finished

Transcriptions are saved as text files in the `transcriptions` directory on the server.

## System Design

- The system implements a pipeline architecture as described in `backend/architecture.md`
- Voice Activity Detection uses the Silero VAD model to identify speech segments
- Noise reduction is performed using the noisereduce package
- The WebSocket connection streams raw audio data to minimize latency

## License

MIT
