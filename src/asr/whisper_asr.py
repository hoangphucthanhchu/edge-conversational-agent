"""Whisper ASR: load model, transcribe audio (file or array) to text."""

from pathlib import Path
from typing import Union

import numpy as np
import whisper


class WhisperASR:
    """Wrapper for OpenAI Whisper (or fine-tuned checkpoint)."""

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        download_root: str | None = None,
    ):
        self.model_name = model_name
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device
        load_name = model_name
        if not Path(model_name).exists() and "/" in model_name:
            # e.g. openai/whisper-small -> small for built-in
            load_name = model_name.split("/")[-1].replace("whisper-", "")
        self._model = whisper.load_model(
            load_name,
            device=device,
            download_root=download_root,
        )

    def transcribe(
        self,
        audio: Union[str, Path, np.ndarray],
        *,
        language: str | None = None,
        **kwargs,
    ) -> str:
        """Transcribe audio to text. Accepts file path or mono float32 array (16kHz)."""
        if isinstance(audio, (str, Path)):
            result = self._model.transcribe(
                str(audio),
                language=language,
                fp16=(self.device == "cuda"),
                **kwargs,
            )
        else:
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32) / (np.iinfo(audio.dtype).max + 1)
            result = self._model.transcribe(
                audio,
                language=language,
                fp16=(self.device == "cuda"),
                **kwargs,
            )
        return (result.get("text") or "").strip()
