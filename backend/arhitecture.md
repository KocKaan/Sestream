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

Audio Chunks → Buffer → Silero VAD → Speech Segments → RNNoise → Whisper → Return Transcription
</rewritten_file>