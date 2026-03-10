"""
Embeddings pipeline using Sentence-Transformers.
Supports multilingual cross-lingual embeddings.
"""
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EmbedPipeline:
    """
    Sentence-Transformers based embedding pipeline.
    Default: paraphrase-multilingual-MiniLM-L12-v2 for cross-lingual support.
    """

    def __init__(self, config: dict):
        self.config = config
        nlp_cfg = config.get("nlp", {})
        self.model_name = nlp_cfg.get(
            "embed_model",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.batch_size = nlp_cfg.get("embedding_batch_size", 64)
        self.max_length = nlp_cfg.get("max_sequence_length", 512)
        self._model = None

    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._model.max_seq_length = self.max_length
            logger.info("Embedding model loaded")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def encode(self, texts: list[str], show_progress: bool = False) -> np.ndarray:
        """Encode a list of texts into embeddings."""
        self._load_model()

        # Clean texts
        clean_texts = [str(t).strip() if t else "" for t in texts]
        # Replace empty texts with placeholder
        clean_texts = [t if t else "empty" for t in clean_texts]

        embeddings = self._model.encode(
            clean_texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def encode_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "pii_masked_text",
        store_col: str = "embedding",
    ) -> pd.DataFrame:
        """Add embeddings to DataFrame."""
        df = df.copy()
        texts = df[text_col].fillna("").tolist()

        logger.info(f"Encoding {len(texts)} texts...")
        embeddings = self.encode(texts, show_progress=True)

        # Store as list of floats (for JSON serialization compatibility)
        df[store_col] = [emb.tolist() for emb in embeddings]
        logger.info(f"Encoding complete: {embeddings.shape}")
        return df

    def get_embeddings_matrix(self, df: pd.DataFrame, col: str = "embedding") -> Optional[np.ndarray]:
        """Extract embedding matrix from DataFrame."""
        if col not in df.columns:
            return None

        vecs = df[col].dropna()
        if len(vecs) == 0:
            return None

        return np.array(vecs.tolist(), dtype=np.float32)

    def detect_language(self, texts: list[str]) -> list[str]:
        """
        Detect language for a list of texts.
        Uses langdetect with fallback to 'en'.
        """
        try:
            from langdetect import detect, LangDetectException

            langs = []
            for text in texts:
                try:
                    lang = detect(str(text)[:500]) if text else "en"
                    langs.append(lang)
                except LangDetectException:
                    langs.append("en")
            return langs

        except ImportError:
            logger.warning("langdetect not available, defaulting to 'en'")
            return ["en"] * len(texts)

    def similarity(self, query: str, texts: list[str]) -> np.ndarray:
        """Compute cosine similarity between query and list of texts."""
        self._load_model()

        all_texts = [query] + texts
        embeddings = self.encode(all_texts)

        query_emb = embeddings[0]
        text_embs = embeddings[1:]

        similarities = np.dot(text_embs, query_emb)
        return similarities
