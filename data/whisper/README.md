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

## Audio

- Format: WAV, 16 kHz mono preferred (script will resample if needed).
- Short clips (e.g. 3–30 s) are fine; long files may be truncated to 30 s.
