"""
ASR pipeline using Whisper for call transcription.
Supports auto language detection, CPU fallback.
"""
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ASRPipeline:
    """
    Whisper-based ASR pipeline.
    - Default: openai/whisper-large-v3 with GPU
    - Fallback: openai/whisper-small.en for CPU
    """

    def __init__(self, config: dict):
        self.config = config
        self.asr_cfg = config.get("asr", {})
        self.model_name = self.asr_cfg.get("model", "openai/whisper-large-v3")
        self.cpu_fallback = self.asr_cfg.get("cpu_fallback", "openai/whisper-small.en")
        self.task = self.asr_cfg.get("task", "transcribe")
        self.language = self.asr_cfg.get("language", None)
        self.use_gpu = self.asr_cfg.get("gpu", True)
        self._pipeline = None
        self._model_loaded = False

    def _load_model(self):
        """Lazy load Whisper model."""
        if self._model_loaded:
            return

        try:
            import torch
            from transformers import pipeline as hf_pipeline

            device = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
            model_to_use = self.model_name if device == "cuda" else self.cpu_fallback

            logger.info(f"Loading ASR model: {model_to_use} on {device}")

            self._pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=model_to_use,
                device=device,
                chunk_length_s=self.asr_cfg.get("chunk_length_s", 30),
                stride_length_s=5,
                return_timestamps=True,
            )
            self._model_loaded = True
            logger.info(f"ASR model loaded: {model_to_use}")

        except Exception as e:
            logger.error(f"Failed to load ASR model: {e}")
            self._pipeline = None

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe an audio file.
        Returns dict with 'text', 'language', 'chunks', 'model'.
        """
        self._load_model()

        if self._pipeline is None:
            return self._fallback_transcribe(audio_path)

        try:
            kwargs = {"task": self.task}
            if self.language:
                kwargs["language"] = self.language

            result = self._pipeline(
                audio_path,
                generate_kwargs=kwargs,
                return_timestamps=True,
            )

            text = result.get("text", "").strip()
            chunks = result.get("chunks", [])
            detected_lang = result.get("language", self.language or "en")

            logger.info(f"Transcribed {audio_path}: {len(text)} chars, lang={detected_lang}")

            return {
                "text": text,
                "language": detected_lang,
                "chunks": chunks,
                "model": self.model_name,
                "audio_path": str(audio_path),
            }

        except Exception as e:
            logger.error(f"Transcription error for {audio_path}: {e}")
            return {
                "text": "",
                "language": "en",
                "chunks": [],
                "model": self.model_name,
                "error": str(e),
                "audio_path": str(audio_path),
            }

    def _fallback_transcribe(self, audio_path: str) -> dict:
        """
        Fallback: try openai-whisper package directly.
        """
        try:
            import whisper

            model_size = "small" if "small" in self.cpu_fallback else "base"
            logger.info(f"Fallback ASR with whisper.{model_size}")

            model = whisper.load_model(model_size)
            result = model.transcribe(
                audio_path,
                task=self.task,
                language=self.language,
            )

            return {
                "text": result.get("text", "").strip(),
                "language": result.get("language", "en"),
                "chunks": [
                    {"timestamp": (s["start"], s["end"]), "text": s["text"]}
                    for s in result.get("segments", [])
                ],
                "model": f"whisper-{model_size}",
                "audio_path": str(audio_path),
            }

        except Exception as e:
            logger.error(f"Fallback transcription failed: {e}")
            return {
                "text": f"[Transcription unavailable: {e}]",
                "language": "en",
                "chunks": [],
                "model": "none",
                "error": str(e),
            }

    def transcribe_dataframe(self, df, audio_col: str = "audio_path") -> "pd.DataFrame":
        """Transcribe all audio files in a DataFrame, adding 'raw_text' and 'lang'."""
        import pandas as pd

        df = df.copy()
        rows_with_audio = df[df[audio_col].notna() & (df[audio_col] != "")]

        for idx, row in rows_with_audio.iterrows():
            result = self.transcribe(str(row[audio_col]))
            df.at[idx, "raw_text"] = result.get("text", "")
            df.at[idx, "lang"] = result.get("language", "en")

        return df

    def compute_wer(self, hypothesis: str, reference: str) -> float:
        """Compute Word Error Rate (WER) for evaluation."""
        hyp_words = hypothesis.lower().split()
        ref_words = reference.lower().split()

        if not ref_words:
            return 0.0

        d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])

        return d[len(ref_words)][len(hyp_words)] / len(ref_words)
