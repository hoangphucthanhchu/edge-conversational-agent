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
    parser.add_argument("--no-freeze-encoder", action="store_false", dest="freeze_encoder", default=True, help="Finetune full model (encoder + decoder). Default: freeze encoder, only finetune decoder.")
    args = parser.parse_args()

    from dataclasses import dataclass
    import torch
    import soundfile as sf
    import librosa
    from datasets import Dataset, DatasetDict
    from transformers import (
        WhisperForConditionalGeneration,
        WhisperProcessor,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
    )
    import evaluate

    # DataCollatorSpeechSeq2SeqWithPadding was removed from transformers; use local implementation (from HF examples).
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        """Collate batch: pad input_features and labels for Whisper seq2seq training."""

        processor: "WhisperProcessor"
        decoder_start_token_id: int
        forward_attention_mask: bool = False

        def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
            model_input_name = self.processor.model_input_names[0]
            input_features = [{model_input_name: f[model_input_name]} for f in features]
            label_features = [{"input_ids": f["labels"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            if self.forward_attention_mask:
                batch["attention_mask"] = torch.LongTensor([f["attention_mask"] for f in features])
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    processor = WhisperProcessor.from_pretrained(args.model, language=None, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)

    if args.freeze_encoder:
        model.freeze_encoder()
        model.model.encoder.gradient_checkpointing = False
        print("Encoder frozen: only decoder will be trained.")
        # Decoder-only: warmup and LR matter; if output is garbage try --no-freeze-encoder or --lr 1e-5

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
        # Keep "path" column; load audio with soundfile in prepare() to avoid torchcodec/FFmpeg.

    default_language = args.dataset_config if args.dataset else "en"

    def prepare(batch):
        path = batch["path"]
        array, sr = sf.read(path, dtype="float32")
        if array.ndim > 1:
            array = array.mean(axis=1)
        if sr != 16000:
            array = librosa.resample(array, orig_sr=sr, target_sr=16000)
        batch["input_features"] = processor.feature_extractor(array, sampling_rate=16000).input_features[0]
        # Labels MUST include the same prefix used at inference (language + task), otherwise
        # the model is trained to predict raw text from BOS but at inference we force
        # <|startoftranscript|><|vi|><|transcribe|><|notimestamps|> first -> distribution shift -> garbage output ("!!!!!!!").
        lang = batch.get("language", default_language)
        prefix_ids = [tid for (_, tid) in processor.get_decoder_prompt_ids(language=lang, task="transcribe")]
        text_ids = processor.tokenizer(batch["sentence"], add_special_tokens=False).input_ids
        end_id = processor.tokenizer.eos_token_id
        if end_id is None:
            end_id = processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")
        batch["labels"] = prefix_ids + text_ids + [end_id]
        return batch

    train_dataset = train_dataset.map(prepare, remove_columns=train_dataset.column_names, num_proc=1)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=False,
    )
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": metric.compute(predictions=pred_str, references=label_str)}

    # Warmup: use ratio so we don't spend 100% of steps in warmup when max_steps is small
    warmup_steps = min(50, max(1, int(0.1 * args.max_steps)))

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
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
        processing_class=processor,
    )
    processor.save_pretrained(args.output_dir)
    trainer.train()
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))
    print("Saved to", args.output_dir)
    print("Update config/pipeline.yaml asr.model_name to:", args.output_dir)


if __name__ == "__main__":
    main()
