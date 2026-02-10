"""Whisper ASR via Hugging Face Transformers. Load model, transcribe audio (file or array) to text."""

import os
from pathlib import Path
from typing import Union

import numpy as np

# Whisper expects 16 kHz mono
SAMPLE_RATE = 16000


def _load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio file to mono float32 at given sample rate."""
    try:
        import librosa
        audio, _ = librosa.load(str(path), sr=sr, mono=True, dtype=np.float32)
        return audio
    except Exception:
        import soundfile as sf
        data, rate = sf.read(str(path), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if rate != sr:
            import librosa
            data = librosa.resample(data, orig_sr=rate, target_sr=sr)
        return data


class WhisperASR:
    """Wrapper for Whisper via Hugging Face Transformers (same stack as train_whisper.py)."""

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        device: str = "auto",
        download_root: str | None = None,
    ):
        import sys
        # Reduce segfault risk on macOS when loading Transformers/PyTorch (OpenMP)
        if sys.platform == "darwin" and "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "1"
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        import torch

        # Backward compat: "base", "small" -> "openai/whisper-base", "openai/whisper-small"
        if model_name in ("tiny", "base", "small", "medium", "large", "large-v2", "large-v3") and "/" not in model_name:
            model_name = f"openai/whisper-{model_name}"
        self.model_name = model_name
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # On macOS (darwin), prefer CPU to avoid MPS/Accelerate segfaults when loading
        if sys.platform == "darwin" and device != "cuda":
            device = "cpu"
        self.device = device

        cache_dir = str(download_root) if download_root else None
        model_path = Path(model_name)
        is_lora_dir = model_path.is_dir() and (model_path / "adapter_config.json").exists()

        if is_lora_dir:
            from peft import PeftModel
            import json
            with open(model_path / "adapter_config.json", encoding="utf-8") as f:
                adapter_cfg = json.load(f)
            base_name = adapter_cfg.get("base_model_name_or_path", "openai/whisper-small")
            self._processor = WhisperProcessor.from_pretrained(
                str(model_path),
                cache_dir=cache_dir,
            )
            self._model = WhisperForConditionalGeneration.from_pretrained(
                base_name,
                cache_dir=cache_dir,
            )
            self._model = PeftModel.from_pretrained(self._model, str(model_path))
        else:
            self._processor = WhisperProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
            )
            self._model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=cache_dir,
            )
        self._model.to(self.device)
        if self.device == "cuda":
            self._model = self._model.half()

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        *,
        language: str | None = None,
        **kwargs,
    ) -> tuple[str, str]:
        """
        Transcribe audio to text. Accepts file path or mono float32 array (16 kHz).
        Returns (text, language_code). If language is None, Whisper auto-detects; language_code is then the detected code (e.g. "vi", "en").
        """
        import torch

        if isinstance(audio, (str, Path)):
            audio = _load_audio(audio)
        else:
            audio = np.asarray(audio, dtype=np.float32)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32) / (np.iinfo(audio.dtype).max + 1)

        inputs = self._processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device)
        if self.device == "cuda":
            input_features = input_features.half()

        # Use language/task in generate() so decoder context matches training (labels with
        # prefix after strip). Do not use forced_decoder_ids for fine-tuned models.
        gen_kwargs = dict(**kwargs)
        if language:
            gen_kwargs["language"] = language
            gen_kwargs["task"] = "transcribe"

        with torch.no_grad():
            # PEFT wrapper expects input as keyword; base Whisper accepts positional input_features
            generated_ids = self._model.generate(
                input_features=input_features,
                **gen_kwargs,
            )

        text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        text = (text or "").strip()

        # Resolved language: forced or detected from first token (index 1 is language token)
        if language:
            resolved_lang = language
        else:
            lang_tokens = self._processor.batch_decode(
                generated_ids[:, 1:2],
                skip_special_tokens=False,
            )
            raw = (lang_tokens[0] or "").strip()
            if raw.startswith("<|") and raw.endswith("|>"):
                resolved_lang = raw[2:-2].strip() or "en"
            else:
                resolved_lang = "en"

        return text, resolved_lang
