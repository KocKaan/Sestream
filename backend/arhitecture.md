# 🧠 Audio Cleaning & Transcription Pipeline Architecture

This document outlines the architecture for a real-time audio streaming system that ingests incoming audio over WebSockets, filters and denoises it, and sends only relevant voice segments to OpenAI Whisper for transcription. The primary goal is **reducing cost and improving accuracy** by removing silence, white noise, and non-human audio before invoking the LLM.

---

## 🔧 Tech Stack

- **Frontend**: WebSocket audio stream (browser/mic)
- **Backend (Python)**:
  - `websockets` / `FastAPI + WebSocket`
  - [`silero-vad`](https://github.com/snakers4/silero-vad): lightweight voice activity detection (VAD)
  - [`rnnoise`](https://github.com/xiph/rnnoise): real-time denoising
  - `ffmpeg` or `pydub`: for audio conversion and chunking
  - `openai` Python SDK: for Whisper transcription

---

## 🧩 System Architecture

```text
[ Mic / Audio Source (Client) ]
            ↓
    [ Audio Chunking (e.g. 1s PCM) ]
            ↓
    [ WebSocket Stream to Backend ]
            ↓
    [ Backend: Python WebSocket Handler ]
            ↓
    [ Silero VAD ]
      ├─ Not speech → Drop ❌
      └─ Is speech → Proceed ✅
            ↓
    [ RNNoise Denoising ]
            ↓
    [ Optional: Convert to WAV/16kHz if needed ]
            ↓
    [ Send to OpenAI Whisper / LLM API ]
            ↓
    [ Return Transcription to Client or Store ]
```

   [WebSocket] → [Audio Buffer] → [Silero VAD] → [Speech Buffer] → [RNNoise] → [Whisper] → [Response]


**Data Types at Each Stage:**\
**WebSocket Input:** Raw bytes (PCM data, typically 16-bit integers)\
**Audio Buffer:** Tensor or NumPy array (float32, normalized to [-1, 1])\
**Silero VAD Output:** Boolean flag indicating speech/non-speech + audio when speech is detected\
**Speech Buffer:** Tensor containing only speech segments\
**RNNoise Input/Output:** Float32 array at 48kHz sample rate\
**Whisper Input:** WAV-formatted audio (can be in-memory)

Key Points About This Data Flow:\
No Intermediate Files\
All data is processed in memory as tensors/arrays\
No file I/O between VAD → RNNoise → Whisper\
Two Types of Buffering

Audio Buffer: Collects incoming audio chunks to form suitable windows for VAD analysis\
Speech Buffer: Accumulates only speech segments for further processing\
Processing Triggered by Events\
Process when buffer reaches minimum size\
Process when speech segment ends (speech → non-speech transition)\
Process after timeout to handle continuous speech\
Data Format Considerations\
Convert between tensor and NumPy array as needed\
Handle resampling for RNNoise (48kHz)\
Normalize audio values for each processing stage\



