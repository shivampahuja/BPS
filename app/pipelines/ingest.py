"""
Ingestion pipeline: CSV/JSON uploads + example loaders for Zendesk/Trustpilot/Qualtrics exports.
Normalizes to canonical interaction schema.
"""
import io
import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

CANONICAL_COLUMNS = [
    "interaction_id",
    "channel",
    "source_system",
    "created_ts",
    "lang",
    "market",
    "author",
    "raw_text",
    "audio_path",
    "pii_masked_text",
    "meta_json",
]

CHANNEL_ALIASES = {
    "call": "call",
    "phone": "call",
    "voice": "call",
    "chat": "chat",
    "live_chat": "chat",
    "livechat": "chat",
    "email": "email",
    "mail": "email",
    "review": "review",
    "trustpilot": "review",
    "google": "review",
    "nps": "nps",
    "survey": "nps",
    "qualtrics": "nps",
    "social": "social",
    "twitter": "social",
    "facebook": "social",
    "zendesk": "ticket",
    "ticket": "ticket",
}

SOURCE_LOADERS = {
    "zendesk": "_load_zendesk",
    "trustpilot": "_load_trustpilot",
    "qualtrics": "_load_qualtrics",
    "generic": "_load_generic",
}


class IngestPipeline:
    """Ingests interaction data from CSV/JSON files and normalizes to canonical schema."""

    def __init__(self, config: dict):
        self.config = config
        self.max_rows = config.get("ui", {}).get("max_upload_rows", 50000)

    def ingest_file(self, file_path: str, source_system: str = "generic") -> pd.DataFrame:
        """Ingest a single file (CSV or JSON) and return canonical DataFrame."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        try:
            if suffix in (".csv", ".tsv"):
                sep = "\t" if suffix == ".tsv" else ","
                raw = pd.read_csv(file_path, sep=sep, nrows=self.max_rows, encoding="utf-8-sig")
            elif suffix == ".json":
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                raw = pd.DataFrame(data if isinstance(data, list) else data.get("data", [data]))
            elif suffix == ".jsonl":
                raw = pd.read_json(file_path, lines=True, nrows=self.max_rows)
            elif suffix in (".xlsx", ".xls"):
                raw = pd.read_excel(file_path, nrows=self.max_rows)
            else:
                raise ValueError(f"Unsupported file format: {suffix}")

            logger.info(f"Loaded {len(raw)} rows from {file_path}")
            return self._normalize(raw, source_system)

        except Exception as e:
            logger.error(f"Error ingesting {file_path}: {e}")
            raise

    def ingest_bytes(self, content: bytes, filename: str, source_system: str = "generic") -> pd.DataFrame:
        """Ingest from bytes (e.g., Gradio upload)."""
        suffix = Path(filename).suffix.lower()
        buf = io.BytesIO(content)

        if suffix in (".csv", ".tsv"):
            sep = "\t" if suffix == ".tsv" else ","
            raw = pd.read_csv(buf, sep=sep, nrows=self.max_rows, encoding="utf-8-sig")
        elif suffix == ".json":
            data = json.load(buf)
            raw = pd.DataFrame(data if isinstance(data, list) else data.get("data", [data]))
        elif suffix == ".jsonl":
            raw = pd.read_json(buf, lines=True, nrows=self.max_rows)
        elif suffix in (".xlsx", ".xls"):
            raw = pd.read_excel(buf, nrows=self.max_rows)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        logger.info(f"Loaded {len(raw)} rows from {filename}")
        return self._normalize(raw, source_system)

    def _normalize(self, raw: pd.DataFrame, source_system: str) -> pd.DataFrame:
        """Normalize arbitrary DataFrame to canonical interaction schema."""
        raw.columns = [c.lower().strip().replace(" ", "_") for c in raw.columns]

        # Detect source system
        if source_system == "auto":
            source_system = self._detect_source(raw.columns.tolist())

        loader_method = getattr(self, SOURCE_LOADERS.get(source_system, "_load_generic"))
        df = loader_method(raw, source_system)

        # Ensure all canonical columns exist
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                df[col] = None

        # Generate interaction IDs for missing
        mask = df["interaction_id"].isna() | (df["interaction_id"] == "")
        df.loc[mask, "interaction_id"] = [
            f"IID-{uuid.uuid4().hex[:12].upper()}" for _ in range(mask.sum())
        ]

        # Normalize timestamps
        if "created_ts" in df.columns:
            df["created_ts"] = pd.to_datetime(df["created_ts"], errors="coerce")
            df["created_ts"] = df["created_ts"].fillna(
                pd.Timestamp.now(tz=timezone.utc)
            )

        # Normalize channels
        if "channel" in df.columns:
            df["channel"] = df["channel"].astype(str).str.lower().map(
                lambda x: CHANNEL_ALIASES.get(x, x)
            )

        # Ensure raw_text is string
        if "raw_text" in df.columns:
            df["raw_text"] = df["raw_text"].fillna("").astype(str)

        # pii_masked_text defaults to raw_text (will be overwritten by PII pipeline)
        df["pii_masked_text"] = df["pii_masked_text"].fillna(df["raw_text"])

        # meta_json: capture leftover columns
        canonical_set = set(CANONICAL_COLUMNS)
        extra_cols = [c for c in df.columns if c not in canonical_set]
        if extra_cols:
            df["meta_json"] = df[extra_cols].apply(
                lambda row: json.dumps(row.dropna().to_dict(), default=str), axis=1
            )

        return df[CANONICAL_COLUMNS].copy()

    def _load_generic(self, raw: pd.DataFrame, source_system: str) -> pd.DataFrame:
        """Generic CSV/JSON mapping with flexible column detection."""
        col_map = {}
        cols = raw.columns.tolist()

        mappings = {
            "interaction_id": ["id", "interaction_id", "ticket_id", "case_id", "conv_id", "conversation_id", "review_id"],
            "channel": ["channel", "type", "source", "contact_type", "medium"],
            "created_ts": ["created_at", "created_ts", "timestamp", "date", "created", "submitted_at", "review_date"],
            "lang": ["lang", "language", "locale"],
            "market": ["market", "country", "region", "geo"],
            "author": ["author", "customer", "customer_id", "user", "reviewer"],
            "raw_text": ["text", "body", "content", "message", "verbatim", "feedback", "comment", "review_text", "raw_text", "transcript"],
            "audio_path": ["audio", "audio_path", "audio_file", "recording"],
            "source_system": ["source_system", "system", "platform"],
        }

        for canonical, aliases in mappings.items():
            for alias in aliases:
                if alias in cols:
                    col_map[canonical] = alias
                    break

        df = pd.DataFrame()
        for canonical, source_col in col_map.items():
            df[canonical] = raw[source_col]

        # Fill source_system
        if "source_system" not in df.columns:
            df["source_system"] = source_system

        return df

    def _load_zendesk(self, raw: pd.DataFrame, source_system: str) -> pd.DataFrame:
        """Zendesk ticket export format."""
        df = pd.DataFrame()
        df["interaction_id"] = raw.get("id", raw.get("ticket_id", None))
        df["channel"] = raw.get("channel", "ticket")
        df["created_ts"] = raw.get("created_at", None)
        df["lang"] = raw.get("locale", None)
        df["market"] = raw.get("custom_field_country", raw.get("organization", None))
        df["author"] = raw.get("requester_name", raw.get("user_name", None))
        df["raw_text"] = raw.get("description", raw.get("body", raw.get("comment", "")))
        df["source_system"] = "zendesk"
        return df

    def _load_trustpilot(self, raw: pd.DataFrame, source_system: str) -> pd.DataFrame:
        """Trustpilot review export format."""
        df = pd.DataFrame()
        df["interaction_id"] = raw.get("review_id", raw.get("id", None))
        df["channel"] = "review"
        df["created_ts"] = raw.get("review_date", raw.get("created_at", None))
        df["lang"] = raw.get("review_language", None)
        df["market"] = raw.get("consumer_country", None)
        df["author"] = raw.get("consumer_display_name", None)
        df["raw_text"] = raw.get("review_body", raw.get("text", ""))
        df["source_system"] = "trustpilot"
        if "review_rating" in raw.columns:
            df["nps_score"] = raw["review_rating"]
        return df

    def _load_qualtrics(self, raw: pd.DataFrame, source_system: str) -> pd.DataFrame:
        """Qualtrics NPS survey export format."""
        df = pd.DataFrame()
        df["interaction_id"] = raw.get("responseid", raw.get("id", None))
        df["channel"] = "nps"
        df["created_ts"] = raw.get("startdate", raw.get("enddate", None))
        df["lang"] = raw.get("userlanguage", raw.get("language", None))
        df["market"] = raw.get("market", raw.get("country", None))
        df["author"] = raw.get("externalreference", raw.get("email", None))

        # Find verbatim/text columns
        text_candidates = [c for c in raw.columns if "comment" in c or "verbatim" in c or "why" in c or "open" in c]
        if text_candidates:
            df["raw_text"] = raw[text_candidates[0]].fillna("")
        else:
            df["raw_text"] = ""

        df["source_system"] = "qualtrics"

        # NPS score
        nps_cols = [c for c in raw.columns if "nps" in c.lower() or "recommend" in c.lower()]
        if nps_cols:
            df["nps_score"] = raw[nps_cols[0]]

        return df

    def _detect_source(self, columns: list) -> str:
        """Auto-detect source system from column names."""
        col_set = set(columns)
        if "review_id" in col_set or "review_body" in col_set:
            return "trustpilot"
        if "ticket_id" in col_set or "requester_name" in col_set:
            return "zendesk"
        if "responseid" in col_set or "userlanguage" in col_set:
            return "qualtrics"
        return "generic"

    def create_sample_interactions(self) -> pd.DataFrame:
        """Create a minimal sample dataset for demo purposes."""
        now = pd.Timestamp.now(tz=timezone.utc)
        samples = [
            {
                "interaction_id": "IID-DEMO-0001",
                "channel": "chat",
                "source_system": "generic",
                "created_ts": now - pd.Timedelta(days=1),
                "lang": "en",
                "market": "UK",
                "author": "customer_001",
                "raw_text": "My device keeps overheating and the battery drains very quickly. This is really frustrating!",
                "audio_path": None,
                "pii_masked_text": None,
                "meta_json": "{}",
            },
            {
                "interaction_id": "IID-DEMO-0002",
                "channel": "call",
                "source_system": "generic",
                "created_ts": now - pd.Timedelta(days=2),
                "lang": "en",
                "market": "US",
                "author": "customer_002",
                "raw_text": "My order was delayed by two weeks! The carrier just kept postponing the delivery date.",
                "audio_path": None,
                "pii_masked_text": None,
                "meta_json": "{}",
            },
            {
                "interaction_id": "IID-DEMO-0003",
                "channel": "review",
                "source_system": "trustpilot",
                "created_ts": now - pd.Timedelta(days=3),
                "lang": "cs",
                "market": "CZ",
                "author": "customer_003",
                "raw_text": "Zařízení se přehřívá a baterie vydrží jen pár hodin. Jsem velmi nespokojený.",
                "audio_path": None,
                "pii_masked_text": None,
                "meta_json": "{}",
            },
        ]
        return pd.DataFrame(samples)
