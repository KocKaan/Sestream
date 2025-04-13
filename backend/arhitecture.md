# 🧠 Audio Cleaning & Transcription Pipeline Architecture

This document outlines the architecture for a real-time audio streaming system that ingests incoming audio over WebSockets, filters and denoises it, and sends only relevant voice segments to OpenAI Whisper for transcription. The primary goal is **reducing cost and improving accuracy** by removing silence, white noise, and non-human audio before invoking the LLM.

---

## 🔧 Tech Stack

- **Frontend**: WebSocket audio stream (browser/mic)
- **Backend (Python)**:
  - `websockets` / `FastAPI + WebSocket`
  - [`silero-vad`](https://github.com/snakers4/silero-vad): lightweight voice activity detection (VAD)
  - [`noisereduce`](https://github.com/timsainb/noisereduce): spectral gating for noise reduction
  - `torch` / `torchaudio`: for audio processing and tensor operations
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
    [ Real-time Audio Buffer ]
            ↓
    [ Silero VAD ]
      ├─ Not speech → Drop ❌
      └─ Is speech → Proceed ✅
            ↓
    [ Speech Buffer ]
            ↓
    [ NoiseReduce Spectral Gating ]
            ↓
    [ Format Conversion for Whisper ]
            ↓
    [ Send to OpenAI Whisper / LLM API ]
            ↓
    [ Return Transcription to Client ]
```

   [WebSocket] → [Audio Buffer] → [Silero VAD] → [Speech Buffer] → [NoiseReduce] → [Whisper] → [Response]


**Data Types at Each Stage:**\
**WebSocket Input:** Raw bytes (PCM data, typically 16-bit integers)\
**Audio Buffer:** Tensor or NumPy array (float32, normalized to [-1, 1])\
**Silero VAD Output:** Boolean flag indicating speech/non-speech + audio when speech is detected\
**Speech Buffer:** Tensor containing only speech segments\
**NoiseReduce Input/Output:** NumPy array (float32, normalized to [-1, 1])\
**Whisper Input:** WAV-formatted audio or direct numpy array (can be in-memory)

---

## 🚀 Real-Time Processing Strategy

### Buffering Strategy
1. **Rolling Audio Buffer**:
   - Maintain a fixed-size rolling buffer (e.g., 10 seconds)
   - Add incoming audio chunks to buffer and drop oldest data when full
   - Process buffer in overlapping windows for VAD detection

2. **Adaptive Speech Segments**:
   - Start capturing when speech is detected
   - Continue capturing until silence threshold is reached
   - Support variable-length speech segments

### Optimization Techniques
1. **Parallel Processing**:
   - Run VAD and noise reduction in separate threads/processes
   - Use `n_jobs` parameter in noisereduce for multi-core processing
   - Buffer outputs to maintain real-time streaming

2. **GPU Acceleration**: 
   - Use TorchGate (from noisereduce.torchgate) for GPU-accelerated denoising
   - Keep models in GPU memory to avoid transfer overhead
   - Batch processing when possible

3. **Chunk-based Processing**:
   - Process audio in fixed-size chunks (e.g., 4096 samples)
   - Overlap chunks slightly to avoid boundary artifacts
   - Use non-stationary noise reduction for dynamic environments

### Latency Management
1. **Tiered Processing**:
   - Fast path: VAD + minimal processing for immediate feedback
   - Full path: Complete noise reduction for final transcription
   - Return preliminary transcriptions while refinements continue

2. **Adaptive Parameters**:
   - Adjust processing parameters based on CPU/GPU load
   - Scale time constants based on observed noise characteristics
   - Increase/decrease chunk sizes based on available resources

3. **Whisper Streaming**:
   - Use OpenAI's streaming endpoint for incremental transcriptions
   - Process overlapping audio chunks to avoid missed words at boundaries
   - Merge transcription results with deduplication logic

---

## 💾 Implementation Considerations

All data is processed in memory as tensors/arrays with no intermediate file I/O between processing stages. The system uses two primary buffers:

1. **Audio Buffer**: Collects incoming audio chunks to form suitable windows for VAD analysis
2. **Speech Buffer**: Accumulates only speech segments for denoising and transcription

Processing is triggered by events:
- When buffer reaches minimum size
- When speech segment ends (speech → non-speech transition)
- After timeout to handle continuous speech

Data format conversions:
- Convert between tensor and NumPy array as needed
- Use non-stationary mode in noisereduce for real-time adaptation to changing noise
- Handle sample rate conversions to match requirements of each component



