"""
PII (Privacy) pipeline using Microsoft Presidio.
Privacy-by-default: all analytics run on masked text.
"""
import logging
import re
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class PIIPipeline:
    """
    Presidio-based PII detection and anonymization.
    Falls back to regex-only mode when Presidio is unavailable.
    """

    def __init__(self, config: dict):
        self.config = config
        self.privacy_cfg = config.get("privacy", {})
        self.policy = self.privacy_cfg.get("pii_policy", "mask")
        self.languages = self.privacy_cfg.get("languages", ["en", "cs"])
        self.entities = self.privacy_cfg.get("entities", [
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
            "LOCATION", "DATE_TIME", "IP_ADDRESS", "URL",
        ])
        self.mask_char = self.privacy_cfg.get("mask_char", "*")
        self.custom_recognizers = self.privacy_cfg.get("custom_recognizers", [])

        self._analyzer = None
        self._anonymizer = None
        self._presidio_available = False
        self._init_presidio()

    def _init_presidio(self):
        """Initialize Presidio with custom recognizers. Graceful degradation."""
        try:
            from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            from presidio_anonymizer import AnonymizerEngine
            from presidio_anonymizer.entities import OperatorConfig
            from presidio_analyzer import PatternRecognizer, Pattern

            # Configure NLP engine
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [
                    {"lang_code": "en", "model_name": "en_core_web_lg"},
                ],
            }

            provider = NlpEngineProvider(nlp_configuration=configuration)
            nlp_engine = provider.create_engine()

            registry = RecognizerRegistry()
            registry.load_predefined_recognizers(nlp_engine=nlp_engine)

            # Add custom recognizers from config
            for recognizer_cfg in self.custom_recognizers:
                try:
                    pattern = Pattern(
                        name=recognizer_cfg["name"],
                        regex=recognizer_cfg["pattern"],
                        score=recognizer_cfg.get("score", 0.8),
                    )
                    custom_rec = PatternRecognizer(
                        supported_entity=recognizer_cfg["name"],
                        patterns=[pattern],
                    )
                    registry.add_recognizer(custom_rec)
                    logger.info(f"Added custom recognizer: {recognizer_cfg['name']}")
                except Exception as e:
                    logger.warning(f"Failed to add custom recognizer {recognizer_cfg.get('name', '?')}: {e}")

            self._analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp_engine)
            self._anonymizer = AnonymizerEngine()
            self._presidio_available = True
            logger.info("Presidio initialized successfully")

        except Exception as e:
            logger.warning(f"Presidio not available, using regex fallback: {e}")
            self._presidio_available = False

    def redact(self, text: str, lang: str = "en") -> tuple[str, list]:
        """
        Redact PII from text.
        Returns (masked_text, pii_audit_log).
        """
        if not text or not isinstance(text, str):
            return text, []

        # Normalize lang for Presidio (cs is not fully supported, fall back to en)
        presidio_lang = "en" if lang not in ["en"] else lang

        if self._presidio_available:
            return self._presidio_redact(text, presidio_lang)
        else:
            return self._regex_redact(text)

    def _presidio_redact(self, text: str, lang: str = "en") -> tuple[str, list]:
        """Presidio-based redaction."""
        try:
            from presidio_anonymizer.entities import OperatorConfig

            results = self._analyzer.analyze(
                text=text,
                language=lang,
                entities=self.entities,
                return_decision_process=False,
            )

            if not results:
                return text, []

            audit_log = [
                {
                    "entity_type": r.entity_type,
                    "start": r.start,
                    "end": r.end,
                    "score": round(r.score, 3),
                    "original": text[r.start:r.end],
                }
                for r in results
            ]

            if self.policy == "mask":
                operators = {
                    entity: OperatorConfig("mask", {"masking_char": self.mask_char, "chars_to_mask": 100, "from_end": False})
                    for entity in self.entities
                }
            elif self.policy == "redact":
                operators = {
                    entity: OperatorConfig("redact")
                    for entity in self.entities
                }
            else:  # pseudonymize
                operators = {
                    entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
                    for entity in self.entities
                }

            anonymized = self._anonymizer.anonymize(
                text=text,
                analyzer_results=results,
                operators=operators,
            )
            return anonymized.text, audit_log

        except Exception as e:
            logger.error(f"Presidio redaction error: {e}")
            return self._regex_redact(text)

    def _regex_redact(self, text: str) -> tuple[str, list]:
        """Fallback regex-based PII redaction."""
        audit_log = []
        patterns = [
            ("EMAIL_ADDRESS", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
            ("PHONE_NUMBER", r"(\+?[0-9]{1,3}[\s\-.]?)?(\(?[0-9]{2,4}\)?[\s\-.]?){2,4}[0-9]{2,4}"),
            ("CREDIT_CARD", r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6011[0-9]{12})\b"),
            ("IP_ADDRESS", r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
            ("URL", r"https?://[^\s]+"),
            ("LOYALTY_ID", r"\b(PM[0-9]{8,12}|LY[A-Z0-9]{8,})\b"),
            ("ORDER_ID", r"\b(ORD[-_][0-9A-Z]{6,12}|#[0-9]{6,10})\b"),
        ]

        masked = text
        for entity_type, pattern in patterns:
            for match in re.finditer(pattern, masked):
                audit_log.append({
                    "entity_type": entity_type,
                    "start": match.start(),
                    "end": match.end(),
                    "score": 0.9,
                    "original": match.group(),
                })

        for entity_type, pattern in patterns:
            replacement = f"<{entity_type}>"
            masked = re.sub(pattern, replacement, masked)

        return masked, audit_log

    def process_dataframe(self, df: pd.DataFrame, text_col: str = "raw_text", lang_col: str = "lang") -> pd.DataFrame:
        """
        Process an entire DataFrame, adding pii_masked_text and pii_audit columns.
        """
        df = df.copy()
        masked_texts = []
        audit_logs = []

        for _, row in df.iterrows():
            text = str(row.get(text_col, "") or "")
            lang = str(row.get(lang_col, "en") or "en")

            masked, audit = self.redact(text, lang)
            masked_texts.append(masked)
            audit_logs.append(audit)

        df["pii_masked_text"] = masked_texts
        df["pii_audit"] = audit_logs

        total_entities = sum(len(a) for a in audit_logs)
        logger.info(
            f"PII processing complete: {len(df)} records, "
            f"{total_entities} entities redacted"
        )
        return df

    def generate_audit_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate a PII audit report from the pii_audit column."""
        if "pii_audit" not in df.columns:
            return pd.DataFrame()

        rows = []
        for _, record in df.iterrows():
            iid = record.get("interaction_id", "?")
            audits = record.get("pii_audit", [])
            if isinstance(audits, list):
                for entry in audits:
                    rows.append({
                        "interaction_id": iid,
                        "entity_type": entry.get("entity_type", "?"),
                        "score": entry.get("score", 0),
                        "original_len": len(entry.get("original", "")),
                    })

        if not rows:
            return pd.DataFrame(columns=["interaction_id", "entity_type", "score", "original_len"])

        return pd.DataFrame(rows)
