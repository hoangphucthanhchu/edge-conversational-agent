"""Piper TTS: text -> WAV (file or bytes). Lightweight, offline."""

from pathlib import Path
from typing import BinaryIO

from piper import PiperVoice


class PiperTTS:
    """Wrapper for Piper TTS. Load .onnx voice, synthesize text to WAV."""

    def __init__(
        self,
        voice: str,
        output_sample_rate: int = 22050,
        use_cuda: bool = False,
    ):
        """
        voice: path to .onnx model (e.g. .../en_US-lessac-medium.onnx),
               or voice id for default cache (e.g. en_US-lessac-medium).
        """
        self.output_sample_rate = output_sample_rate
        self._use_cuda = use_cuda
        self._voice_path = self._resolve_voice(voice)
        self._voice = PiperVoice.load(self._voice_path, use_cuda=use_cuda)

    def set_voice(self, voice: str) -> None:
        """Switch to another voice (e.g. by ASR language)."""
        self._voice_path = self._resolve_voice(voice)
        self._voice = PiperVoice.load(self._voice_path, use_cuda=self._use_cuda)

    def _resolve_voice(self, voice: str) -> str:
        p = Path(voice)
        if p.exists():
            return str(p)
        name = voice if voice.endswith(".onnx") else f"{voice}.onnx"
        voice_id = voice.removesuffix(".onnx") if voice.endswith(".onnx") else voice
        project_root = Path(__file__).resolve().parent.parent.parent
        for base in [
            Path.home() / ".local" / "share" / "piper",
            Path.cwd() / "voices",
            project_root / "voices",
            project_root,  # e.g. en_US-lessac-medium.onnx in repo root
        ]:
            candidate = base / name
            if candidate.exists():
                return str(candidate)
        # Try to download the voice into project voices/
        try:
            from piper.download_voices import download_voice
            download_dir = project_root / "voices"
            download_dir.mkdir(parents=True, exist_ok=True)
            download_voice(voice_id, download_dir)
            candidate = download_dir / name
            if candidate.exists():
                return str(candidate)
        except Exception as e:
            raise FileNotFoundError(
                f"Piper voice not found: {voice}. Tried to download but failed: {e}. "
                "Install the voice manually: python -m piper.download_voices <voice_id>"
            ) from e
        raise FileNotFoundError(
            f"Piper voice not found: {voice}. "
            "Run: python -m piper.download_voices <voice_id> and set path in config."
        )

    def synthesize_to_file(self, text: str, wav_path: str | Path) -> None:
        """Synthesize text to a WAV file."""
        import wave
        path = Path(wav_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to WAV bytes (for latency measurement or in-memory use)."""
        import io
        import wave
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            self._voice.synthesize_wav(text, wav_file)
        return buf.getvalue()
