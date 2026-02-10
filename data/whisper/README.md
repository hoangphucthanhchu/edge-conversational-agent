# Whisper fine-tune data (VN + EN)

Place VN+EN audio and transcripts here for fine-tuning.

## Manifest format

Use a manifest CSV or JSON with columns:

- **path**: relative path to WAV file (from this directory) or absolute path
- **text**: transcript (Vietnamese or English)
- **language**: `vi` or `en`

Example `manifest.csv`:

```csv
path,text,language
clips/sample_vi_001.wav,Xin chào đây là mẫu tiếng Việt.,vi
clips/sample_en_001.wav,This is a sample English sentence.,en
```

Or load from Hugging Face: e.g. `mozilla-foundation/common_voice_17_0` with config `vi` and `en` (see `scripts/train_whisper.py --dataset`).

## VIVOS subset (30–60 min)

If using [VIVOS](https://www.kaggle.com/datasets/kynthesis/vivos-vietnamese-speech-corpus-for-asr) (Vietnamese speech corpus ~15h):

1. Place the data under `vivos/train/` (with `prompts.txt` and `waves/`).
2. Build a 30–60 min subset and manifest:
   ```bash
   python scripts/build_vivos_subset.py --min-min 30 --max-min 60
   ```
   Output: `data/whisper/vivos_subset_manifest.csv`.
3. Fine-tune **decoder-only**, 1–2 epochs:
   ```bash
   python scripts/train_whisper.py \
     --manifest data/whisper/vivos_subset_manifest.csv \
     --num-train-epochs 2 \
     --lr 1e-5 \
     --output-dir models/whisper-small-vivos
   ```
   By default the encoder is frozen (only the decoder is trained). `--lr` is learning rate (default 3e-5). Omit `--num-train-epochs` and use `--max-steps` if you want to train by step count.

## Audio

- Format: WAV, 16 kHz mono preferred (script will resample if needed).
- Short clips (e.g. 3–30 s) are fine; long files may be truncated to 30 s.
