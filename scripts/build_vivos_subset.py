#!/usr/bin/env python3
"""
Build a 30–60 minute subset from VIVOS train and write manifest CSV for Whisper fine-tuning.
Usage:
  python scripts/build_vivos_subset.py
  python scripts/build_vivos_subset.py --min-min 30 --max-min 60 --out data/whisper/vivos_subset_manifest.csv
"""

import argparse
import csv
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_prompts(prompts_path: Path) -> list[tuple[str, str]]:
    """Parse VIVOS prompts.txt: each line is 'UTTERANCE_ID TRANSCRIPT'. Returns [(utt_id, text), ...]."""
    rows = []
    with open(prompts_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            utt_id = parts[0]
            text = parts[1] if len(parts) > 1 else ""
            rows.append((utt_id, text))
    return rows


def find_wav_path(utt_id: str, waves_dir: Path) -> Path | None:
    """VIVOS: utt_id is like VIVOSSPK01_R001 -> waves/VIVOSSPK01/VIVOSSPK01_R001.wav."""
    # Speaker prefix: VIVOSSPK01_R001 -> VIVOSSPK01
    if "_" in utt_id:
        speaker = utt_id.split("_", 1)[0]
    else:
        return None
    wav = waves_dir / speaker / f"{utt_id}.wav"
    return wav if wav.exists() else None


def get_duration_seconds(wav_path: Path) -> float:
    import wave
    try:
        with wave.open(str(wav_path), "rb") as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        try:
            import soundfile as sf
            info = sf.info(wav_path)
            return len(info) / info.samplerate
        except Exception:
            return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VIVOS 30–60 min subset manifest.")
    parser.add_argument("--vivos-dir", type=Path, default=PROJECT_ROOT / "data" / "whisper" / "vivos" / "train", help="VIVOS train directory (prompts.txt + waves/)")
    parser.add_argument("--min-min", type=float, default=30, help="Minimum subset duration in minutes")
    parser.add_argument("--max-min", type=float, default=60, help="Maximum subset duration in minutes")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "whisper" / "vivos_subset_manifest.csv", help="Output manifest CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--list-ids", type=Path, default=None, help="Optional: write selected utterance IDs to this file")
    args = parser.parse_args()

    prompts_path = args.vivos_dir / "prompts.txt"
    waves_dir = args.vivos_dir / "waves"
    if not prompts_path.exists():
        print(f"Not found: {prompts_path}", file=sys.stderr)
        sys.exit(1)
    if not waves_dir.is_dir():
        print(f"Not found: {waves_dir}", file=sys.stderr)
        sys.exit(1)

    rows = parse_prompts(prompts_path)
    # Build (utt_id, text, wav_path) and get duration for each
    candidates = []
    for utt_id, text in rows:
        wav = find_wav_path(utt_id, waves_dir)
        if wav is None:
            continue
        dur = get_duration_seconds(wav)
        if dur <= 0:
            continue
        candidates.append((utt_id, text.strip(), wav, dur))

    if not candidates:
        print("No valid (utt_id, wav) pairs found.", file=sys.stderr)
        sys.exit(1)

    total_all = sum(d for (_, _, _, d) in candidates)
    print(f"VIVOS train: {len(candidates)} utterances, total {total_all / 60:.1f} min")

    # Shuffle and greedily add until we are in [min_min, max_min]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    target_min = (args.min_min + args.max_min) / 2
    target_sec = target_min * 60
    selected = []
    total_sec = 0.0
    for utt_id, text, wav, dur in candidates:
        if total_sec >= args.max_min * 60:
            break
        selected.append((utt_id, text, wav, dur))
        total_sec += dur
        if total_sec >= args.min_min * 60 and total_sec <= args.max_min * 60:
            break
    # If we're still under min, add until we hit min (or run out)
    if total_sec < args.min_min * 60:
        for utt_id, text, wav, dur in candidates:
            if (utt_id, text, wav, dur) in [(s[0], s[1], s[2], s[3]) for s in selected]:
                continue
            selected.append((utt_id, text, wav, dur))
            total_sec += dur
            if total_sec >= args.min_min * 60:
                break

    total_sec = sum(s[3] for s in selected)
    print(f"Subset: {len(selected)} utterances, {total_sec / 60:.1f} min")

    # Manifest: path relative to data/whisper so that load_local_manifest(data_dir=PROJECT_ROOT/"data"/"whisper") works
    data_whisper = PROJECT_ROOT / "data" / "whisper"
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "text", "language"])
        writer.writeheader()
        for utt_id, text, wav, _ in selected:
            try:
                rel = wav.relative_to(data_whisper)
            except ValueError:
                rel = wav
            writer.writerow({"path": str(rel), "text": text, "language": "vi"})

    print(f"Wrote manifest: {out_path}")

    if args.list_ids:
        args.list_ids.parent.mkdir(parents=True, exist_ok=True)
        with open(args.list_ids, "w", encoding="utf-8") as f:
            for utt_id, _, _, _ in selected:
                f.write(utt_id + "\n")
        print(f"Wrote utterance IDs: {args.list_ids}")


if __name__ == "__main__":
    main()
