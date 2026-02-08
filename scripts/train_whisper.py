#!/usr/bin/env python3
"""Fine-tune Whisper (VN + EN) with Hugging Face Transformers. Saves to models/whisper-small-vien."""

import argparse
import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def load_local_manifest(manifest_path: Path, data_dir: Path) -> "list[dict]":
    """Load manifest CSV/JSON: path, text, language. Resolve paths relative to data_dir."""
    rows = []
    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        with open(manifest_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                path = r.get("path", "").strip()
                if not path:
                    continue
                if not Path(path).is_absolute():
                    path = str(data_dir / path)
                rows.append({"path": path, "sentence": r.get("text", r.get("sentence", "")), "language": r.get("language", "en")})
    elif suffix == ".json":
        import json
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data if isinstance(data, list) else data.get("data", []):
            path = item.get("path", "")
            if not Path(path).is_absolute():
                path = str(data_dir / path)
            rows.append({"path": path, "sentence": item.get("text", item.get("sentence", "")), "language": item.get("language", "en")})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Whisper for VN+EN.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "whisper" / "manifest.csv", help="Manifest CSV/JSON (path, text, language)")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "whisper", help="Base dir for relative paths in manifest")
    parser.add_argument("--model", type=str, default="openai/whisper-small", help="Base Whisper model")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "models" / "whisper-small-vien", help="Save checkpoint here")
    parser.add_argument("--max-steps", type=int, default=500, help="Training steps (small demo)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset name (e.g. mozilla-foundation/common_voice_17_0) to use instead of manifest")
    parser.add_argument("--dataset-config", type=str, default="vi", help="Dataset config/subset (e.g. vi, en)")
    args = parser.parse_args()

    from datasets import Dataset, DatasetDict, Audio
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    from transformers import DataCollatorSpeechSeq2SeqWithPadding
    import evaluate

    processor = WhisperProcessor.from_pretrained(args.model, language=None, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)

    if args.dataset:
        from datasets import load_dataset
        common_voice = load_dataset(args.dataset, args.dataset_config, split="train+validation", trust_remote_code=True)
        if "sentence" not in common_voice.column_names and "text" in common_voice.column_names:
            common_voice = common_voice.rename_column("text", "sentence")
        common_voice = common_voice.remove_columns([c for c in common_voice.column_names if c not in ("audio", "sentence")])
        common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
        train_dataset = common_voice
    else:
        if not args.manifest.exists():
            print(f"Manifest not found: {args.manifest}. Create data/whisper/manifest.csv or use --dataset.", file=sys.stderr)
            sys.exit(1)
        rows = load_local_manifest(args.manifest, args.data_dir)
        if not rows:
            print("No rows in manifest.", file=sys.stderr)
            sys.exit(1)
        train_dataset = Dataset.from_list(rows)
        train_dataset = train_dataset.cast_column("path", Audio(sampling_rate=16000))
        if "path" in train_dataset.column_names:
            train_dataset = train_dataset.rename_column("path", "audio")

    def prepare(batch):
        audio = batch["audio"]
        if isinstance(audio, dict):
            batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        else:
            batch["input_features"] = processor.feature_extractor(batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"]).input_features[0]
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        return batch

    train_dataset = train_dataset.map(prepare, remove_columns=train_dataset.column_names, num_proc=1)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": metric.compute(predictions=pred_str, references=label_str)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_steps=50,
        fp16=True,
        report_to="none",
        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,
    )
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    processor.save_pretrained(args.output_dir)
    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    print("Saved to", args.output_dir)
    print("Update config/pipeline.yaml asr.model_name to:", args.output_dir)


if __name__ == "__main__":
    main()
