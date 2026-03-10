"""
Sentiment analysis pipeline - multilingual with DSAT flagging.
Supports sentiment ∈ {neg, neu, pos}, score ∈ [-1, 1], and dsat_flag.
"""
import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Language-specific sentiment lexicons for override/adjustment
SENTIMENT_LEXICONS = {
    "en": {
        "strong_negative": [
            "terrible", "awful", "horrible", "disgusting", "outrageous",
            "unacceptable", "furious", "worst", "useless", "broken",
            "scam", "fraud", "incompetent", "disaster", "nightmare"
        ],
        "negative": [
            "bad", "poor", "slow", "late", "wrong", "failed",
            "disappointed", "frustrated", "annoyed", "issue", "problem",
            "error", "broken", "not working", "refused", "delayed"
        ],
        "positive": [
            "great", "excellent", "amazing", "fantastic", "wonderful",
            "helpful", "fast", "resolved", "satisfied", "happy", "love",
            "perfect", "brilliant", "outstanding", "impressed"
        ],
        "strong_positive": [
            "exceptional", "superb", "phenomenal", "life-changing",
            "incredible", "best ever", "highly recommend", "five stars"
        ],
    },
    "cs": {
        "strong_negative": [
            "hrozné", "strašné", "hnusné", "skandální", "nepřijatelné",
            "nejhorší", "zbytečné", "rozbitý", "podvod", "neschopný",
            "katastrofa", "noční můra", "zoufalý"
        ],
        "negative": [
            "špatný", "pomalý", "pozdní", "chybný", "selhal",
            "zklamaný", "frustrovaný", "problém", "chyba", "nefunguje",
            "odmítnutý", "zpožděný", "nevyřešený"
        ],
        "positive": [
            "skvělý", "výborný", "úžasný", "fantastický", "nádherný",
            "nápomocný", "rychlý", "vyřešený", "spokojený", "šťastný",
            "perfektní", "bezchybný", "nadšený"
        ],
        "strong_positive": [
            "výjimečný", "fenomenální", "nejlepší", "doporučuji",
            "pět hvězd", "naprosto spokojen"
        ],
    },
}

DSAT_COMPLAINT_KEYWORDS = {
    "en": [
        "complaint", "refund", "escalate", "manager", "lawsuit",
        "never buying again", "waste of money", "rip off", "fraud",
        "terrible service", "unacceptable", "demand refund"
    ],
    "cs": [
        "stížnost", "vrácení peněz", "eskalovat", "vedoucí",
        "žaloba", "nikdy víc", "ztráta peněz", "podvod",
        "hrozná obsluha", "nepřijatelné", "požaduji vrácení"
    ],
}


class SentimentPipeline:
    """
    Multilingual sentiment analysis pipeline.
    Uses transformer model with lexicon overlays per language.
    Falls back to lexicon-only analysis if model unavailable.
    """

    def __init__(self, config: dict):
        self.config = config
        nlp_cfg = config.get("nlp", {})
        self.model_name = nlp_cfg.get(
            "sentiment_model",
            "cardiffnlp/twitter-xlm-roberta-base-sentiment"
        )
        self.dsat_cfg = config.get("analytics", {}).get("dsat", {})
        self.sentiment_threshold = self.dsat_cfg.get("sentiment_threshold", -0.3)

        self._classifier = None
        self._model_loaded = False

    def _load_model(self):
        """Lazy load sentiment model."""
        if self._model_loaded:
            return

        try:
            from transformers import pipeline as hf_pipeline

            logger.info(f"Loading sentiment model: {self.model_name}")
            self._classifier = hf_pipeline(
                "sentiment-analysis",
                model=self.model_name,
                top_k=None,
                truncation=True,
                max_length=512,
            )
            self._model_loaded = True
            logger.info("Sentiment model loaded")

        except Exception as e:
            logger.warning(f"Sentiment model unavailable, using lexicon baseline: {e}")
            self._model_loaded = True  # Don't retry

    def analyze(self, text: str, lang: str = "en") -> dict:
        """
        Analyze sentiment for a single text.
        Returns: {sentiment, score, dsat_flag, method}
        """
        if not text or not isinstance(text, str) or not text.strip():
            return {"sentiment": "neu", "score": 0.0, "dsat_flag": False, "method": "empty"}

        self._load_model()

        if self._classifier is not None:
            result = self._model_analyze(text, lang)
        else:
            result = self._lexicon_analyze(text, lang)

        # Apply DSAT rules
        result["dsat_flag"] = self._check_dsat(text, lang, result["score"])
        return result

    def _model_analyze(self, text: str, lang: str) -> dict:
        """Transformer model-based sentiment analysis."""
        try:
            truncated = text[:512]
            outputs = self._classifier(truncated)

            if isinstance(outputs, list) and outputs:
                if isinstance(outputs[0], list):
                    outputs = outputs[0]

                # Map to standard labels
                label_map = {
                    "POSITIVE": "pos",
                    "NEGATIVE": "neg",
                    "NEUTRAL": "neu",
                    "positive": "pos",
                    "negative": "neg",
                    "neutral": "neu",
                    "LABEL_0": "neg",
                    "LABEL_1": "neu",
                    "LABEL_2": "pos",
                }

                best = max(outputs, key=lambda x: x["score"])
                sentiment = label_map.get(best["label"], "neu")
                confidence = best["score"]

                # Convert to [-1, 1] score
                score_map = {"neg": -confidence, "neu": 0.0, "pos": confidence}
                score = score_map.get(sentiment, 0.0)

                # Apply lexicon adjustment
                lex_score = self._lexicon_score(text, lang)
                final_score = 0.7 * score + 0.3 * lex_score
                final_sentiment = self._score_to_label(final_score)

                return {
                    "sentiment": final_sentiment,
                    "score": round(final_score, 4),
                    "confidence": round(confidence, 4),
                    "method": "model+lexicon",
                }

        except Exception as e:
            logger.error(f"Model sentiment failed: {e}")

        return self._lexicon_analyze(text, lang)

    def _lexicon_analyze(self, text: str, lang: str) -> dict:
        """Lexicon-only sentiment analysis as baseline."""
        score = self._lexicon_score(text, lang)
        return {
            "sentiment": self._score_to_label(score),
            "score": round(score, 4),
            "confidence": 0.6,
            "method": "lexicon",
        }

    def _lexicon_score(self, text: str, lang: str) -> float:
        """Compute a sentiment score from lexicon."""
        text_lower = text.lower()
        lexicon = SENTIMENT_LEXICONS.get(lang, SENTIMENT_LEXICONS.get("en", {}))

        score = 0.0
        hit_count = 0

        weights = {
            "strong_positive": 1.0,
            "positive": 0.5,
            "negative": -0.5,
            "strong_negative": -1.0,
        }

        for category, weight in weights.items():
            for term in lexicon.get(category, []):
                if term.lower() in text_lower:
                    score += weight
                    hit_count += 1

        if hit_count == 0:
            return 0.0

        # Normalize to [-1, 1]
        normalized = max(-1.0, min(1.0, score / max(hit_count, 1)))
        return normalized

    def _score_to_label(self, score: float) -> str:
        """Convert numeric score to label."""
        if score < -0.15:
            return "neg"
        elif score > 0.15:
            return "pos"
        return "neu"

    def _check_dsat(self, text: str, lang: str, score: float) -> bool:
        """Check if interaction is DSAT (dissatisfied)."""
        if score < self.sentiment_threshold:
            return True

        text_lower = text.lower()
        keywords = DSAT_COMPLAINT_KEYWORDS.get(lang, DSAT_COMPLAINT_KEYWORDS.get("en", []))
        for kw in keywords:
            if kw.lower() in text_lower:
                return True

        return False

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "pii_masked_text",
        lang_col: str = "lang",
    ) -> pd.DataFrame:
        """Process entire DataFrame, adding sentiment columns."""
        df = df.copy()
        sentiments = []
        scores = []
        dsat_flags = []
        methods = []

        for _, row in df.iterrows():
            text = str(row.get(text_col, "") or "")
            lang = str(row.get(lang_col, "en") or "en")

            result = self.analyze(text, lang)
            sentiments.append(result["sentiment"])
            scores.append(result["score"])
            dsat_flags.append(result["dsat_flag"])
            methods.append(result["method"])

        df["sentiment"] = sentiments
        df["sentiment_score"] = scores
        df["dsat_flag"] = dsat_flags
        df["sentiment_method"] = methods

        logger.info(
            f"Sentiment complete: "
            f"pos={df['sentiment'].eq('pos').sum()}, "
            f"neu={df['sentiment'].eq('neu').sum()}, "
            f"neg={df['sentiment'].eq('neg').sum()}, "
            f"dsat={df['dsat_flag'].sum()}"
        )
        return df

    def get_sentiment_distribution(self, df: pd.DataFrame) -> dict:
        """Get sentiment distribution stats."""
        if "sentiment" not in df.columns:
            return {}

        total = len(df)
        return {
            "positive": int(df["sentiment"].eq("pos").sum()),
            "neutral": int(df["sentiment"].eq("neu").sum()),
            "negative": int(df["sentiment"].eq("neg").sum()),
            "dsat": int(df.get("dsat_flag", pd.Series(dtype=bool)).sum()),
            "avg_score": float(df["sentiment_score"].mean()) if "sentiment_score" in df else 0.0,
            "total": total,
        }

    def compute_metrics(
        self,
        y_true: list[str],
        y_pred: list[str],
    ) -> dict:
        """Compute precision, recall, F1 per class for evaluation."""
        try:
            from sklearn.metrics import classification_report

            report = classification_report(
                y_true, y_pred,
                labels=["neg", "neu", "pos"],
                output_dict=True,
            )
            return report
        except Exception as e:
            logger.error(f"Metrics computation failed: {e}")
            return {}
