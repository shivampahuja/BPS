"""
Compliance pipeline.
Flags: authentication, adverse_event, health_mention, age_verification,
nicotine_declaration, privacy/GDPR.
"""
import logging
import re
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


COMPLIANCE_PATTERNS = {
    "authentication": {
        "en": [
            r"(verify|verification|authenticate|authentication)",
            r"(date of birth|security question|confirm.{0,20}identity|who am i speaking with)",
            r"(account.{0,20}(verify|confirm|check))",
        ],
        "cs": [
            r"(ověřit|ověření|autentizovat|autentizace)",
            r"(datum.{0,10}narozen|bezpečnostní otázka|totožnost|s kým mluvím)",
        ],
    },
    "adverse_event": {
        "en": [
            r"\b(health|medical|doctor|hospital|clinic|injury|hurt|pain|illness|sick|disease)\b",
            r"\b(adverse|side effect|reaction|allergy|allergic|symptom)\b",
            r"\b(emergency|ambulance|911|medical attention)\b",
        ],
        "cs": [
            r"\b(zdraví|lékař|nemocnice|klinika|zranění|bolest|nemoc|onemocnění)\b",
            r"\b(nežádoucí|vedlejší účinek|reakce|alergie|alergický|příznaky)\b",
            r"\b(záchranná služba|pohotovost|lékařská pomoc)\b",
        ],
    },
    "health_mention": {
        "en": [
            r"\b(health|medical|physical|body|chest|heart|breath|dizzy|nausea)\b",
            r"\b(symptoms?|condition|diagnos|prescription|medication)\b",
        ],
        "cs": [
            r"\b(zdraví|lékařský|fyzický|tělo|hrudník|srdce|dech|závrať|nevolnost)\b",
            r"\b(příznaky|stav|diagnóza|předpis|lék)\b",
        ],
    },
    "age_verification": {
        "en": [
            r"\b(age.{0,15}(verify|check|confirm|restrict)|verify.{0,15}age)\b",
            r"\b(18\+|21\+|underage|minor|adult.{0,10}(confirm|verify))\b",
            r"\b(age.{0,10}requirement|date of birth|born in)\b",
        ],
        "cs": [
            r"\b(věk.{0,15}(ověřit|zkontrolovat|potvrdit|omezení))\b",
            r"\b(nezletilý|plnoletý|18 let|21 let|věková hranice)\b",
            r"\b(datum.{0,10}narozen|rok.{0,10}narozen)\b",
        ],
    },
    "nicotine_declaration": {
        "en": [
            r"\b(nicotine|tobacco|smoking|vaping|e.?cigarette|vape|pod)\b",
            r"\b(nicotine.{0,20}(content|level|warning|declaration|confirm))\b",
        ],
        "cs": [
            r"\b(nikotin|tabák|kouření|vapování|elektronická.{0,10}cigareta)\b",
            r"\b(nikotin.{0,20}(obsah|úroveň|varování|prohlášení|potvrdit))\b",
        ],
    },
    "privacy": {
        "en": [
            r"\b(privacy|personal data|data protection|GDPR|opt.?out|delete.{0,15}data)\b",
            r"\b(right.{0,20}forgotten|data.{0,10}(share|usage|request)|consent)\b",
        ],
        "cs": [
            r"\b(soukromí|osobní údaje|ochrana dat|GDPR|odhlásit|smazat.{0,15}data)\b",
            r"\b(právo.{0,20}zapomenut|sdílení dat|souhlas)\b",
        ],
    },
    "gdpr": {
        "en": [
            r"\b(GDPR|data access.{0,20}request|subject access|right to access|SAR)\b",
            r"\b(delete my data|erase data|data erasure|portability)\b",
        ],
        "cs": [
            r"\b(GDPR|žádost.{0,20}přístup.{0,20}data|právo.{0,20}přístupu|SAR)\b",
            r"\b(smazat.{0,10}data|vymazat.{0,10}data|přenositelnost)\b",
        ],
    },
}

SEVERITY_MAP = {
    "adverse_event": "critical",
    "health_mention": "critical",
    "authentication": "high",
    "age_verification": "high",
    "nicotine_declaration": "high",
    "privacy": "high",
    "gdpr": "high",
}


class CompliancePipeline:
    """
    Compliance flag detection pipeline.
    Flags critical/high severity compliance events for routing.
    """

    FLAG_TYPES = list(COMPLIANCE_PATTERNS.keys())

    def __init__(self, config: dict):
        self.config = config
        self._compiled = {}

        for flag_type, lang_patterns in COMPLIANCE_PATTERNS.items():
            self._compiled[flag_type] = {}
            for lang, patterns in lang_patterns.items():
                self._compiled[flag_type][lang] = [
                    re.compile(p, re.IGNORECASE) for p in patterns
                ]

    def detect(self, text: str, lang: str = "en") -> dict:
        """
        Detect compliance flags in a single text.
        Returns: {flag_type: bool, ...}
        """
        if not text or not isinstance(text, str):
            return {f: False for f in self.FLAG_TYPES}

        lang_key = lang if lang in ["en", "cs"] else "en"
        results = {}

        for flag_type in self.FLAG_TYPES:
            patterns = (
                self._compiled[flag_type].get(lang_key, []) +
                self._compiled[flag_type].get("en", [])
            )
            detected = any(p.search(text) for p in patterns)
            results[flag_type] = detected

        return results

    def detect_with_severity(self, text: str, lang: str = "en") -> list:
        """
        Detect flags and return list of {flag_type, severity, evidence} dicts.
        """
        if not text or not isinstance(text, str):
            return []

        lang_key = lang if lang in ["en", "cs"] else "en"
        flags = []

        for flag_type in self.FLAG_TYPES:
            patterns = (
                self._compiled[flag_type].get(lang_key, []) +
                self._compiled[flag_type].get("en", [])
            )
            for p in patterns:
                match = p.search(text)
                if match:
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    flags.append({
                        "flag_type": flag_type,
                        "severity": SEVERITY_MAP.get(flag_type, "medium"),
                        "evidence": f"...{text[start:end]}...",
                        "pattern": p.pattern,
                    })
                    break  # One flag per type

        return flags

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "pii_masked_text",
        lang_col: str = "lang",
    ) -> pd.DataFrame:
        """Add compliance flag columns to DataFrame."""
        df = df.copy()

        flag_data: dict[str, list] = {f: [] for f in self.FLAG_TYPES}
        all_flags_lists = []
        critical_flags_lists = []

        for _, row in df.iterrows():
            text = str(row.get(text_col, "") or "")
            lang = str(row.get(lang_col, "en") or "en")

            detected = self.detect(text, lang)

            for f in self.FLAG_TYPES:
                flag_data[f].append(detected[f])

            flags_list = [f for f, v in detected.items() if v]
            critical_list = [
                f for f in flags_list
                if SEVERITY_MAP.get(f, "medium") == "critical"
            ]
            all_flags_lists.append(flags_list)
            critical_flags_lists.append(critical_list)

        for f in self.FLAG_TYPES:
            df[f"compliance_{f}"] = flag_data[f]

        df["compliance_flags"] = all_flags_lists
        df["critical_flags"] = critical_flags_lists
        df["has_critical_flag"] = df["critical_flags"].apply(lambda x: len(x) > 0)
        df["has_compliance_flag"] = df["compliance_flags"].apply(lambda x: len(x) > 0)

        critical_count = df["has_critical_flag"].sum()
        logger.info(
            f"Compliance complete: "
            f"{df['has_compliance_flag'].sum()} interactions with flags, "
            f"{critical_count} with critical flags"
        )
        return df

    def get_flag_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get compliance flag summary."""
        rows = []
        for f in self.FLAG_TYPES:
            col = f"compliance_{f}"
            if col in df.columns:
                count = int(df[col].sum())
                rate = float(df[col].mean()) * 100
                rows.append({
                    "flag_type": f,
                    "severity": SEVERITY_MAP.get(f, "medium"),
                    "count": count,
                    "rate_pct": round(rate, 2),
                })

        return pd.DataFrame(rows).sort_values("count", ascending=False)

    def get_critical_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return interactions with critical compliance flags."""
        if "has_critical_flag" not in df.columns:
            return pd.DataFrame()
        return df[df["has_critical_flag"]].copy()

    def compute_metrics(
        self,
        y_true: dict,
        y_pred: dict,
    ) -> dict:
        """Compute precision/recall/F1 for compliance detection."""
        try:
            from sklearn.metrics import precision_recall_fscore_support

            metrics = {}
            for flag_type in self.FLAG_TYPES:
                true_labels = y_true.get(flag_type, [])
                pred_labels = y_pred.get(flag_type, [])

                if not true_labels or not pred_labels:
                    continue

                p, r, f1, _ = precision_recall_fscore_support(
                    true_labels, pred_labels, average="binary", zero_division=0
                )
                metrics[flag_type] = {
                    "precision": round(float(p), 4),
                    "recall": round(float(r), 4),
                    "f1": round(float(f1), 4),
                    "severity": SEVERITY_MAP.get(flag_type, "medium"),
                }

            return metrics

        except Exception as e:
            logger.error(f"Compliance metrics failed: {e}")
            return {}
