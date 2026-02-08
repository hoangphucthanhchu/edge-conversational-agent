# Edge Conversational Agent

Local pipeline: **Audio → ASR (Whisper) → RAG → LLM → TTS**, with end-to-end latency measurement (mock edge).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

- **Ollama**: Install and run [Ollama](https://ollama.ai), then `ollama pull llama3.2`.
- **Piper TTS**: Voice models are downloaded on first use (or place `.onnx` + `.json` in config).

## Run demo

```bash
python scripts/run_demo.py path/to/audio.wav
```

Optional: `MOCK_EDGE=1 python scripts/run_demo.py audio.wav` to simulate edge (reduced threads).

## Fine-tune Whisper (VN + EN)

1. Place VN+EN audio + transcripts in `data/whisper/` (see `scripts/train_whisper.py` for format).
2. Run: `python scripts/train_whisper.py`
3. Update `config/pipeline.yaml` → `asr.model_name` to your saved model path (e.g. `models/whisper-small-vien`).

## Project layout

- `config/pipeline.yaml` – model paths, RAG/LLM/TTS config
- `src/asr/` – Whisper ASR (Hugging Face Transformers; same stack as fine-tuning)
- `src/rag/` – RAG (ChromaDB + sentence-transformers)
- `src/llm/` – Ollama client
- `src/tts/` – Piper TTS
- `src/pipeline.py` – orchestration
- `src/latency.py` – per-stage and E2E latency
- `scripts/run_demo.py` – demo + latency report
- `scripts/train_whisper.py` – Whisper fine-tuning (VN+EN); inference uses same HF checkpoint
