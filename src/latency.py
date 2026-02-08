"""Per-stage and E2E latency measurement (mock edge)."""

from dataclasses import dataclass
from typing import Callable


@dataclass
class LatencyReport:
    """Latency in ms per stage and E2E. RTF = E2E / audio_duration_sec."""

    asr_ms: float = 0.0
    rag_ms: float = 0.0
    llm_ms: float = 0.0
    tts_ms: float = 0.0
    e2e_ms: float = 0.0
    rtf: float | None = None  # real-time factor

    def __str__(self) -> str:
        lines = [
            "Stage   Latency (ms)",
            f"ASR     {self.asr_ms:.0f}",
            f"RAG     {self.rag_ms:.0f}",
            f"LLM     {self.llm_ms:.0f}",
            f"TTS     {self.tts_ms:.0f}",
            "---",
            f"E2E     {self.e2e_ms:.0f} ms",
        ]
        if self.rtf is not None:
            lines.append(f"RTF     {self.rtf:.2f}")
        return "\n".join(lines)


def measure_latency(
    run_fn: Callable[[], tuple[tuple[str, str, str, bytes, str], LatencyReport]],
    audio_duration_sec: float | None = None,
) -> tuple[tuple[str, str, str, bytes, str], LatencyReport]:
    """
    Run the pipeline via run_fn() which returns (result, report) with per-stage timings.
    Optionally set report.rtf from audio_duration_sec.
    """
    (result, report) = run_fn()
    if audio_duration_sec and audio_duration_sec > 0 and report.e2e_ms > 0:
        report.rtf = (report.e2e_ms / 1000) / audio_duration_sec
    return result, report
