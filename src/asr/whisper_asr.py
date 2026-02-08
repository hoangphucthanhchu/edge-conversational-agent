"""Whisper ASR via Hugging Face Transformers. Load model, transcribe audio (file or array) to text."""

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
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        import torch

        # Backward compat: "base", "small" -> "openai/whisper-base", "openai/whisper-small"
        if model_name in ("tiny", "base", "small", "medium", "large", "large-v2", "large-v3") and "/" not in model_name:
            model_name = f"openai/whisper-{model_name}"
        self.model_name = model_name
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        cache_dir = str(download_root) if download_root else None
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
    ) -> str:
        """Transcribe audio to text. Accepts file path or mono float32 array (16 kHz)."""
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

        forced_decoder_ids = None
        if language:
            forced_decoder_ids = self._processor.get_decoder_prompt_ids(
                language=language,
                task="transcribe",
            )

        with torch.no_grad():
            generated_ids = self._model.generate(
                input_features,
                forced_decoder_ids=forced_decoder_ids,
                **kwargs,
            )

        text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
        return (text or "").strip()
