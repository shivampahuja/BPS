"""
Topic modeling pipeline using BERTopic with guided seeds from PMI taxonomy.
Supports hierarchical reduction and multilingual cross-lingual clustering.
"""
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class TopicsPipeline:
    """
    BERTopic-based topic modeling with PMI taxonomy seeds.
    """

    def __init__(self, config: dict, taxonomy_path: Optional[str] = None):
        self.config = config
        self.nlp_cfg = config.get("nlp", {})
        self.topic_cfg = self.nlp_cfg.get("topic", {})
        self.taxonomy_cfg = config.get("taxonomy", {})

        self.model_path = Path(
            config.get("storage", {}).get("bertopic_path", "models/bertopic_model")
        )
        self.taxonomy_version = "1.0.0"
        self.taxonomy = {}
        self.seed_topic_list = []
        self.theme_labels = {}

        if taxonomy_path:
            self._load_taxonomy(taxonomy_path)

        self._topic_model = None

    def _load_taxonomy(self, taxonomy_path: str):
        """Load PMI taxonomy and extract seed words."""
        try:
            with open(taxonomy_path, "r", encoding="utf-8") as f:
                tax_data = yaml.safe_load(f)

            self.taxonomy = tax_data.get("themes", {})
            self.taxonomy_version = tax_data.get("taxonomy_version", "1.0.0")

            # Build seed topic list for guided BERTopic
            self.seed_topic_list = []
            self.theme_labels = {}

            for theme_key, theme_data in self.taxonomy.items():
                seeds_en = theme_data.get("seeds", {}).get("en", [])
                seeds_cs = theme_data.get("seeds", {}).get("cs", [])
                all_seeds = seeds_en + seeds_cs

                if all_seeds:
                    self.seed_topic_list.append(all_seeds)

                self.theme_labels[theme_key] = {
                    "en": theme_data.get("labels", {}).get("en", theme_key),
                    "cs": theme_data.get("labels", {}).get("cs", theme_key),
                    "level1": theme_data.get("level1", ""),
                    "level2": theme_data.get("level2", ""),
                    "level3": theme_data.get("level3", ""),
                }

            logger.info(
                f"Loaded taxonomy: {len(self.taxonomy)} themes, "
                f"{len(self.seed_topic_list)} seed groups"
            )

        except Exception as e:
            logger.error(f"Failed to load taxonomy: {e}")

    def train(
        self,
        texts: list[str],
        embeddings: Optional[np.ndarray] = None,
        language: str = "multilingual",
    ) -> "BERTopic":
        """Train BERTopic model on texts."""
        try:
            from bertopic import BERTopic
            from sklearn.feature_extraction.text import CountVectorizer
            from umap import UMAP
            from hdbscan import HDBSCAN

            min_topic_size = self.topic_cfg.get("min_topic_size", 5)
            nr_topics = self.topic_cfg.get("nr_topics", "auto")
            guided = self.topic_cfg.get("guided", True)
            low_memory = self.topic_cfg.get("low_memory", False)

            umap_model = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                low_memory=low_memory,
                random_state=42,
            )

            hdbscan_model = HDBSCAN(
                min_cluster_size=max(3, min_topic_size),
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )

            vectorizer_model = CountVectorizer(
                ngram_range=(1, 2),
                stop_words=None,
                min_df=2,
                max_features=10000,
            )

            seed_topics = self.seed_topic_list if guided and self.seed_topic_list else None

            model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                language=language,
                seed_topic_list=seed_topics,
                nr_topics=nr_topics,
                low_memory=low_memory,
                verbose=False,
                calculate_probabilities=True,
            )

            logger.info(f"Training BERTopic on {len(texts)} texts...")

            if embeddings is not None:
                topics, probs = model.fit_transform(texts, embeddings=embeddings)
            else:
                topics, probs = model.fit_transform(texts)

            self._topic_model = model
            logger.info(f"Training complete: {len(model.get_topic_info())} topics found")

            return model

        except Exception as e:
            logger.error(f"BERTopic training failed: {e}")
            return self._fallback_topic_model(texts)

    def _fallback_topic_model(self, texts: list[str]):
        """Simple TF-IDF based fallback for topic modeling."""
        logger.info("Using TF-IDF fallback for topics")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans

            vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
            X = vectorizer.fit_transform(texts)

            n_clusters = min(20, max(3, len(texts) // 10))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            topics = kmeans.fit_predict(X)

            return {"type": "fallback_kmeans", "topics": topics.tolist(), "n_clusters": n_clusters}

        except Exception as e:
            logger.error(f"Fallback topic model failed: {e}")
            return None

    def predict(self, texts: list[str], embeddings: Optional[np.ndarray] = None) -> tuple:
        """Predict topics for new texts."""
        if self._topic_model is None:
            logger.warning("Topic model not trained, returning -1 topics")
            return [-1] * len(texts), [None] * len(texts)

        try:
            if embeddings is not None:
                topics, probs = self._topic_model.transform(texts, embeddings=embeddings)
            else:
                topics, probs = self._topic_model.transform(texts)
            return topics, probs
        except Exception as e:
            logger.error(f"Topic prediction failed: {e}")
            return [-1] * len(texts), [None] * len(texts)

    def get_topic_info(self) -> pd.DataFrame:
        """Get topic information DataFrame."""
        if self._topic_model is None:
            return pd.DataFrame()

        try:
            info = self._topic_model.get_topic_info()
            # Map to taxonomy labels where possible
            info["taxonomy_label"] = info["Name"].apply(
                lambda x: self._match_to_taxonomy(x)
            )
            return info
        except Exception as e:
            logger.error(f"get_topic_info failed: {e}")
            return pd.DataFrame()

    def _match_to_taxonomy(self, topic_label: str) -> str:
        """Try to match a BERTopic label to PMI taxonomy."""
        if not topic_label or not self.theme_labels:
            return topic_label

        topic_words = set(topic_label.lower().replace("_", " ").split())

        best_match = None
        best_score = 0

        for theme_key, theme_data in self.theme_labels.items():
            label_words = set(theme_data.get("en", "").lower().split())
            overlap = len(topic_words & label_words)
            if overlap > best_score:
                best_score = overlap
                best_match = theme_data.get("en", theme_key)

        return best_match or topic_label

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "pii_masked_text",
        embed_col: str = "embedding",
    ) -> pd.DataFrame:
        """Add topic assignments to DataFrame."""
        df = df.copy()
        texts = df[text_col].fillna("").tolist()
        embeddings = None

        if embed_col in df.columns:
            try:
                embeddings = np.array(df[embed_col].tolist(), dtype=np.float32)
            except Exception:
                embeddings = None

        if self._topic_model is None:
            logger.info("Training new topic model...")
            self.train(texts, embeddings)

        topics, probs = self.predict(texts, embeddings)
        df["topic_id"] = topics
        df["topic_prob"] = [float(p[0]) if p is not None and len(p) > 0 else 0.0 for p in (probs if probs is not None else [[0]] * len(topics))]

        # Add taxonomy labels
        topic_info = self.get_topic_info()
        if not topic_info.empty and "Topic" in topic_info.columns:
            topic_label_map = dict(zip(topic_info["Topic"], topic_info["Name"]))
            taxonomy_map = dict(zip(topic_info["Topic"], topic_info.get("taxonomy_label", topic_info["Name"])))
            df["topic_label"] = df["topic_id"].map(topic_label_map).fillna("Unknown")
            df["taxonomy_label"] = df["topic_id"].map(taxonomy_map).fillna("Unknown")
        else:
            df["topic_label"] = "Unknown"
            df["taxonomy_label"] = "Unknown"

        # Map to PMI taxonomy levels using seed-based matching
        df["l1_domain"] = df["taxonomy_label"].apply(self._get_level1)
        df["l2_theme"] = df["taxonomy_label"].apply(self._get_level2)

        logger.info(f"Topic modeling complete: {df['topic_id'].nunique()} unique topics")
        return df

    def _get_level1(self, label: str) -> str:
        """Get level1 domain from taxonomy label."""
        for theme_key, theme_data in self.theme_labels.items():
            if theme_data.get("en", "").lower() in label.lower():
                return theme_data.get("level1", "")
        return ""

    def _get_level2(self, label: str) -> str:
        """Get level2 theme from taxonomy label."""
        for theme_key, theme_data in self.theme_labels.items():
            if theme_data.get("en", "").lower() in label.lower():
                return theme_data.get("level2", "")
        return ""

    def save(self, path: Optional[str] = None):
        """Save trained topic model."""
        if self._topic_model is None:
            logger.warning("No topic model to save")
            return

        save_path = Path(path or self.model_path)
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            self._topic_model.save(str(save_path))
            logger.info(f"Topic model saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save topic model: {e}")

    def load(self, path: Optional[str] = None):
        """Load saved topic model."""
        try:
            from bertopic import BERTopic

            load_path = Path(path or self.model_path)
            if load_path.exists():
                self._topic_model = BERTopic.load(str(load_path))
                logger.info(f"Topic model loaded from {load_path}")
            else:
                logger.warning(f"No topic model found at {load_path}")
        except Exception as e:
            logger.error(f"Failed to load topic model: {e}")

    def get_representative_docs(self, topic_id: int, n: int = 5) -> list:
        """Get representative documents for a topic."""
        if self._topic_model is None:
            return []
        try:
            return self._topic_model.get_representative_docs(topic_id)[:n]
        except Exception:
            return []

    def get_topics_for_display(self, lang: str = "en") -> pd.DataFrame:
        """Get topic info formatted for display in UI."""
        topic_info = self.get_topic_info()
        if topic_info.empty:
            return pd.DataFrame()

        rows = []
        for _, row in topic_info.iterrows():
            topic_id = row.get("Topic", -1)
            if topic_id == -1:
                continue

            # Try to match to PMI taxonomy
            tax_label = row.get("taxonomy_label", row.get("Name", f"Topic {topic_id}"))

            # Find level info
            level1 = ""
            level2 = ""
            level3 = ""
            labels_cs = ""

            for theme_key, theme_data in self.theme_labels.items():
                en_label = theme_data.get("en", "")
                if en_label and en_label.lower() in (tax_label or "").lower():
                    level1 = theme_data.get("level1", "")
                    level2 = theme_data.get("level2", "")
                    level3 = theme_data.get("level3", "")
                    labels_cs = theme_data.get("cs", "")
                    break

            rows.append({
                "topic_id": topic_id,
                "label_en": tax_label or f"Topic {topic_id}",
                "label_cs": labels_cs,
                "level1": level1,
                "level2": level2,
                "level3": level3,
                "count": row.get("Count", 0),
                "representation": row.get("Representation", ""),
            })

        return pd.DataFrame(rows)
