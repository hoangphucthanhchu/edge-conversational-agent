"""Orchestrate: audio -> ASR -> RAG -> LLM -> TTS. Config from config/pipeline.yaml."""

import time
from pathlib import Path
from typing import Any

import yaml

from src.asr.whisper_asr import WhisperASR
from src.latency import LatencyReport
from src.llm.ollama_client import OllamaClient
from src.rag.retriever import RAGRetriever
from src.tts.piper_tts import PiperTTS


def load_config(config_path: str | Path = "config/pipeline.yaml") -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Pipeline:
    """Audio -> ASR -> RAG -> LLM -> TTS. Returns (transcript, context, answer, audio_bytes), LatencyReport."""

    def __init__(self, config: dict[str, Any] | None = None, config_path: str | Path = "config/pipeline.yaml"):
        cfg = config or load_config(config_path)
        asr_cfg = cfg.get("asr") or {}
        rag_cfg = cfg.get("rag") or {}
        llm_cfg = cfg.get("llm") or {}
        tts_cfg = cfg.get("tts") or {}

        # language: None or "auto" or "" => Whisper auto-detect; "vi", "en", etc. => force language
        _lang = asr_cfg.get("language")
        self._asr_language = None if _lang in (None, "auto", "") else str(_lang).strip() or None
        self._current_language: str | None = None  # set after run() from config or ASR-detected

        self.asr = WhisperASR(
            model_name=asr_cfg.get("model_name", "base"),
            device=asr_cfg.get("device", "auto"),
        )
        self.rag = RAGRetriever(
            embed_model=rag_cfg.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2"),
            persist_dir=rag_cfg.get("persist_dir", "data/rag/faiss_index"),
            top_k=rag_cfg.get("top_k", 3),
            chunk_size=rag_cfg.get("chunk_size", 512),
            chunk_overlap=rag_cfg.get("chunk_overlap", 64),
        )
        docs_dir = rag_cfg.get("docs_dir") or "data/rag"
        self.rag.build_index(docs_dir)

        self.llm = OllamaClient(
            model=llm_cfg.get("model", "llama3.2"),
            base_url=llm_cfg.get("base_url", "http://localhost:11434"),
        )
        self._tts_default_voice = tts_cfg.get("voice", "en_US-lessac-medium")
        self._tts_voices_by_language = tts_cfg.get("voices_by_language") or {}
        self.tts = PiperTTS(
            voice=self._tts_default_voice,
            output_sample_rate=tts_cfg.get("output_sample_rate", 22050),
        )

    def run(self, audio: str | Path | bytes) -> tuple[tuple[str, str, str, bytes, str], LatencyReport]:
        """
        Run pipeline: audio -> transcript -> context -> answer -> audio_bytes.
        If asr.language is null in config, Whisper auto-detects; resolved language is set and used for the rest of the flow.
        Returns ((transcript, context, answer, audio_bytes, language), LatencyReport).
        """
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        transcript, language = self.asr.transcribe(audio, language=self._asr_language)
        t_asr = time.perf_counter()
        # language from config (if forced) or Whisper-detected; use for rest of flow
        self._current_language = language

        context = self.rag.retrieve(transcript)
        t_rag = time.perf_counter()

        answer = self.llm.generate(query=transcript, context=context)
        t_llm = time.perf_counter()

        # Chọn TTS voice theo ngôn ngữ ASR
        voice = self._tts_voices_by_language.get(language) if language else None
        if voice:
            self.tts.set_voice(voice)
        else:
            self.tts.set_voice(self._tts_default_voice)

        audio_bytes = self.tts.synthesize(answer) if answer.strip() else b""
        t_tts = time.perf_counter()

        report = LatencyReport(
            asr_ms=(t_asr - t0) * 1000,
            rag_ms=(t_rag - t_asr) * 1000,
            llm_ms=(t_llm - t_rag) * 1000,
            tts_ms=(t_tts - t_llm) * 1000,
            e2e_ms=(t_tts - t_start) * 1000,
        )
        return (transcript, context, answer, audio_bytes, language), report
