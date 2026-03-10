"""
Semantic search and RAG pipeline.
Uses FAISS for vector search, optional Haystack for natural-language Q&A.
Cross-lingual: English queries can retrieve Czech documents.
"""
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Semantic search over PII-masked interaction texts.
    Returns interaction_id citations for retrieved documents.
    """

    def __init__(self, config: dict, embed_pipeline=None):
        self.config = config
        self.embed_pipeline = embed_pipeline
        storage_cfg = config.get("storage", {})
        self.index_path = storage_cfg.get("index_path", "models/vector_store/index")

        self._index = None
        self._documents: list[dict] = []
        self._interaction_ids: list[str] = []
        self._faiss_available = False
        self._check_faiss()

    def _check_faiss(self):
        """Check if FAISS is available."""
        try:
            import faiss
            self._faiss_available = True
        except ImportError:
            logger.warning("FAISS not available; falling back to brute-force search")

    def build_index(self, df: pd.DataFrame, text_col: str = "pii_masked_text") -> bool:
        """Build FAISS index from DataFrame."""
        if df.empty:
            return False

        texts = df[text_col].fillna("").tolist()
        interaction_ids = df.get("interaction_id", pd.Series(range(len(df)))).fillna("").tolist()

        if self.embed_pipeline is None:
            logger.warning("No embedding pipeline, cannot build index")
            return False

        logger.info(f"Building vector index for {len(texts)} documents...")
        embeddings = self.embed_pipeline.encode(texts)

        self._documents = [
            {
                "interaction_id": str(iid),
                "text": text,
                "idx": i,
            }
            for i, (iid, text) in enumerate(zip(interaction_ids, texts))
        ]
        self._interaction_ids = [str(iid) for iid in interaction_ids]

        if self._faiss_available:
            self._build_faiss_index(embeddings)
        else:
            # Store embeddings for brute-force search
            self._embeddings_matrix = embeddings

        logger.info("Vector index built successfully")
        return True

    def _build_faiss_index(self, embeddings: np.ndarray):
        """Build FAISS flat L2 index."""
        import faiss

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product for normalized vecs
        index.add(embeddings.astype(np.float32))
        self._index = index
        logger.info(f"FAISS index built: {index.ntotal} vectors, dim={dim}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        lang: Optional[str] = None,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        Semantic search for query.
        Returns list of {interaction_id, text, score, rank}.
        """
        if not query or self.embed_pipeline is None:
            return []

        if not self._documents:
            return []

        query_emb = self.embed_pipeline.encode([query])

        if self._faiss_available and self._index is not None:
            return self._faiss_search(query_emb, top_k)
        elif hasattr(self, "_embeddings_matrix"):
            return self._brute_force_search(query_emb, top_k)
        else:
            logger.warning("No index available for search")
            return []

    def _faiss_search(self, query_emb: np.ndarray, top_k: int) -> list[dict]:
        """FAISS inner product search."""
        scores, indices = self._index.search(
            query_emb.astype(np.float32), min(top_k, len(self._documents))
        )

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self._documents):
                continue
            doc = self._documents[idx]
            results.append({
                "interaction_id": doc["interaction_id"],
                "text": doc["text"],
                "score": float(score),
                "rank": rank + 1,
            })
        return results

    def _brute_force_search(self, query_emb: np.ndarray, top_k: int) -> list[dict]:
        """Brute-force cosine similarity search."""
        scores = np.dot(self._embeddings_matrix, query_emb[0])
        top_indices = np.argsort(-scores)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            doc = self._documents[idx]
            results.append({
                "interaction_id": doc["interaction_id"],
                "text": doc["text"],
                "score": float(scores[idx]),
                "rank": rank + 1,
            })
        return results

    def answer_question(
        self,
        question: str,
        top_k: int = 5,
        context_window: int = 10,
    ) -> dict:
        """
        Answer a natural language question using retrieved context.
        Returns: {answer, sources: [interaction_id, ...], context}
        """
        retrieved = self.search(question, top_k=context_window)

        if not retrieved:
            return {
                "answer": "No relevant interactions found.",
                "sources": [],
                "context": "",
            }

        # Try Haystack if available
        try:
            return self._haystack_qa(question, retrieved, top_k)
        except Exception:
            pass

        # Fallback: extractive summary
        return self._extractive_answer(question, retrieved, top_k)

    def _haystack_qa(
        self,
        question: str,
        retrieved: list[dict],
        top_k: int,
    ) -> dict:
        """Haystack-based extractive QA."""
        from haystack.nodes import FARMReader
        from haystack.schema import Document

        docs = [
            Document(content=r["text"], id=r["interaction_id"])
            for r in retrieved
        ]

        reader = FARMReader(
            model_name_or_path="deepset/roberta-base-squad2",
            use_gpu=False,
        )
        predictions = reader.predict(question=question, documents=docs, top_k=top_k)

        answer_text = predictions["answers"][0].answer if predictions["answers"] else "No answer found"
        sources = list({a.document_id for a in predictions["answers"][:top_k]})

        return {
            "answer": answer_text,
            "sources": sources,
            "context": "\n---\n".join(r["text"][:300] for r in retrieved[:top_k]),
        }

    def _extractive_answer(
        self,
        question: str,
        retrieved: list[dict],
        top_k: int,
    ) -> dict:
        """Simple extractive fallback: return top retrieved text snippets."""
        sources = [r["interaction_id"] for r in retrieved[:top_k]]
        context = "\n---\n".join(
            f"[{r['interaction_id']}] {r['text'][:300]}" for r in retrieved[:top_k]
        )

        return {
            "answer": f"Found {len(retrieved)} relevant interactions. "
                      f"Top match (ID: {retrieved[0]['interaction_id']}): "
                      f"{retrieved[0]['text'][:200]}...",
            "sources": sources,
            "context": context,
        }

    def save_index(self, path: Optional[str] = None):
        """Save FAISS index and document store."""
        save_path = Path(path or self.index_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if self._faiss_available and self._index is not None:
            import faiss
            faiss.write_index(self._index, str(save_path) + ".faiss")

        # Save document metadata
        with open(str(save_path) + "_docs.json", "w") as f:
            json.dump(self._documents, f)

        logger.info(f"Index saved to {save_path}")

    def load_index(self, path: Optional[str] = None):
        """Load FAISS index and document store."""
        load_path = Path(path or self.index_path)

        if self._faiss_available:
            faiss_path = str(load_path) + ".faiss"
            if Path(faiss_path).exists():
                import faiss
                self._index = faiss.read_index(faiss_path)

        docs_path = str(load_path) + "_docs.json"
        if Path(docs_path).exists():
            with open(docs_path, "r") as f:
                self._documents = json.load(f)
            self._interaction_ids = [d["interaction_id"] for d in self._documents]

        logger.info(f"Index loaded from {load_path}")
