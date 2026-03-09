"""
Behavior detection pipeline.
Detects: empathy, authentication, resolution_confirmation, escalation, adverse_event.
Pattern-based with configurable patterns per language.
"""
import logging
import re
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class BehaviorsPipeline:
    """
    Detects CII behavioral signals in interaction text.
    Uses configurable regex patterns per language.
    """

    BEHAVIOR_TYPES = [
        "empathy",
        "authentication",
        "resolution_confirmation",
        "escalation",
        "adverse_event",
    ]

    def __init__(self, config: dict):
        self.config = config
        behavior_cfg = config.get("analytics", {}).get("behaviors", {})

        # Build compiled pattern sets
        self._patterns: dict[str, dict[str, list]] = {}
        for behavior in self.BEHAVIOR_TYPES:
            patterns_en = behavior_cfg.get(behavior, {}).get("patterns_en", [])
            patterns_cs = behavior_cfg.get(behavior, {}).get("patterns_cs", [])
            self._patterns[behavior] = {
                "en": [re.compile(p, re.IGNORECASE) for p in patterns_en],
                "cs": [re.compile(p, re.IGNORECASE) for p in patterns_cs],
            }

    def detect(self, text: str, lang: str = "en") -> dict:
        """
        Detect all behaviors in a single text.
        Returns dict: {behavior_type: bool, ...}
        """
        if not text or not isinstance(text, str):
            return {b: False for b in self.BEHAVIOR_TYPES}

        lang_key = lang if lang in ["en", "cs"] else "en"
        results = {}

        for behavior in self.BEHAVIOR_TYPES:
            patterns = (
                self._patterns[behavior].get(lang_key, []) +
                self._patterns[behavior].get("en", [])  # Always check EN too
            )
            detected = any(p.search(text) for p in patterns)
            results[behavior] = detected

        return results

    def detect_with_evidence(self, text: str, lang: str = "en") -> dict:
        """
        Detect behaviors and return matching evidence snippets.
        """
        if not text or not isinstance(text, str):
            return {}

        lang_key = lang if lang in ["en", "cs"] else "en"
        results = {}

        for behavior in self.BEHAVIOR_TYPES:
            patterns = (
                self._patterns[behavior].get(lang_key, []) +
                self._patterns[behavior].get("en", [])
            )
            evidence = []
            for p in patterns:
                match = p.search(text)
                if match:
                    start = max(0, match.start() - 20)
                    end = min(len(text), match.end() + 20)
                    evidence.append(f"...{text[start:end]}...")

            results[behavior] = {
                "detected": bool(evidence),
                "evidence": evidence[:3],  # Max 3 examples
            }

        return results

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "pii_masked_text",
        lang_col: str = "lang",
    ) -> pd.DataFrame:
        """Add behavior detection columns to DataFrame."""
        df = df.copy()

        behavior_data: dict[str, list] = {b: [] for b in self.BEHAVIOR_TYPES}
        all_behaviors_lists = []

        for _, row in df.iterrows():
            text = str(row.get(text_col, "") or "")
            lang = str(row.get(lang_col, "en") or "en")

            detected = self.detect(text, lang)

            for b in self.BEHAVIOR_TYPES:
                behavior_data[b].append(detected[b])

            # Create list of detected behaviors
            behaviors_list = [b for b, v in detected.items() if v]
            all_behaviors_lists.append(behaviors_list)

        for b in self.BEHAVIOR_TYPES:
            df[f"behavior_{b}"] = behavior_data[b]

        df["behaviors"] = all_behaviors_lists

        # Log frequencies
        for b in self.BEHAVIOR_TYPES:
            freq = df[f"behavior_{b}"].mean() * 100
            logger.info(f"Behavior '{b}': {freq:.1f}%")

        return df

    def get_frequency_report(
        self,
        df: pd.DataFrame,
        group_by: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Get behavior frequency report, optionally grouped by queue/vendor/market.
        """
        behavior_cols = [f"behavior_{b}" for b in self.BEHAVIOR_TYPES if f"behavior_{b}" in df.columns]
        if not behavior_cols:
            return pd.DataFrame()

        if group_by:
            valid_groups = [g for g in group_by if g in df.columns]
            if valid_groups:
                grouped = df.groupby(valid_groups)[behavior_cols].agg(
                    ["sum", "mean"]
                )
                grouped.columns = [f"{col[0]}_{col[1]}" for col in grouped.columns]
                return grouped.reset_index()

        # Overall frequencies
        freq = {}
        for col in behavior_cols:
            b = col.replace("behavior_", "")
            freq[b] = {
                "count": int(df[col].sum()),
                "rate": float(df[col].mean()),
                "percentage": round(float(df[col].mean()) * 100, 2),
            }

        return pd.DataFrame(freq).T.reset_index().rename(columns={"index": "behavior"})

    def compute_metrics(
        self,
        y_true: dict,
        y_pred: dict,
    ) -> dict:
        """Compute precision, recall, F1 for behavior detection."""
        try:
            from sklearn.metrics import precision_recall_fscore_support

            metrics = {}
            for behavior in self.BEHAVIOR_TYPES:
                true_labels = y_true.get(behavior, [])
                pred_labels = y_pred.get(behavior, [])

                if not true_labels or not pred_labels:
                    continue

                p, r, f1, _ = precision_recall_fscore_support(
                    true_labels, pred_labels, average="binary", zero_division=0
                )
                metrics[behavior] = {
                    "precision": round(float(p), 4),
                    "recall": round(float(r), 4),
                    "f1": round(float(f1), 4),
                }

            return metrics

        except Exception as e:
            logger.error(f"Behavior metrics failed: {e}")
            return {}
