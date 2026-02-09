#!/usr/bin/env python3
"""Demo pipeline: audio -> ASR -> RAG -> LLM -> TTS. Print latency table (mock edge)."""

import argparse
import os
import sys
from pathlib import Path

# Run from project root so src is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Reduce segfault risk when loading Whisper on macOS (OpenMP / MPS)
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

from src.latency import measure_latency
from src.pipeline import Pipeline


def get_audio_duration_sec(audio_path: str | Path) -> float:
    try:
        import soundfile as sf
        data, rate = sf.read(str(audio_path))
        return len(data) / float(rate)
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Edge Conversational Agent demo with latency report.")
    parser.add_argument("audio", type=Path, help="Path to input audio file (WAV)")
    parser.add_argument("-n", "--runs", type=int, default=1, help="Number of runs for averaging (default 1)")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "pipeline.yaml", help="Pipeline config YAML")
    args = parser.parse_args()

    if not args.audio.exists():
        print(f"Error: audio file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    if os.environ.get("MOCK_EDGE") == "1":
        os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")

    pipeline = Pipeline(config_path=args.config)
    audio_duration_sec = get_audio_duration_sec(args.audio)

    def run_once() -> tuple[tuple[str, str, str, bytes, str], object]:
        return pipeline.run(args.audio)

    runs = max(1, args.runs)
    reports = []
    for i in range(runs):
        (result, report) = measure_latency(run_once, audio_duration_sec)
        transcript, context, answer, audio_bytes, language = result
        reports.append(report)
        if i == 0:
            print("Language:", language or "(none)")
            print("Transcript:", transcript or "(empty)")
            print("Answer:", answer or "(empty)")
            if audio_bytes:
                out_path = args.audio.parent / (args.audio.stem + "_out.wav")
                out_path.write_bytes(audio_bytes)
                print("Output audio:", out_path)

    if runs == 1:
        print("\n" + str(reports[0]))
    else:
        import statistics
        r = reports[0]
        r.asr_ms = statistics.mean((x.asr_ms for x in reports))
        r.rag_ms = statistics.mean((x.rag_ms for x in reports))
        r.llm_ms = statistics.mean((x.llm_ms for x in reports))
        r.tts_ms = statistics.mean((x.tts_ms for x in reports))
        r.e2e_ms = statistics.mean((x.e2e_ms for x in reports))
        if reports[0].rtf is not None:
            r.rtf = statistics.mean((x.rtf for x in reports if x.rtf is not None))
        print("\nLatency (mean over {} runs):\n{}".format(runs, r))


if __name__ == "__main__":
    main()
