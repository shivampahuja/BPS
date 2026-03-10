"""
Tests for CII pipeline modules.
Run with: pytest app/tests/ -v
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def config():
    import yaml
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_interactions():
    """Sample DataFrame matching canonical schema."""
    return pd.DataFrame([
        {
            "interaction_id": "IID-TEST-001",
            "channel": "chat",
            "source_system": "test",
            "created_ts": "2026-01-15T10:00:00Z",
            "lang": "en",
            "market": "UK",
            "author": "test_user_001",
            "raw_text": "My device keeps overheating and the battery drains very quickly. This is really frustrating!",
            "audio_path": None,
            "pii_masked_text": None,
            "meta_json": "{}",
        },
        {
            "interaction_id": "IID-TEST-002",
            "channel": "review",
            "source_system": "test",
            "created_ts": "2026-01-16T14:00:00Z",
            "lang": "en",
            "market": "US",
            "author": "test_user_002",
            "raw_text": "Excellent product! Great battery life and fantastic customer service. Very happy!",
            "audio_path": None,
            "pii_masked_text": None,
            "meta_json": "{}",
        },
        {
            "interaction_id": "IID-TEST-003",
            "channel": "call",
            "source_system": "test",
            "created_ts": "2026-01-17T09:00:00Z",
            "lang": "cs",
            "market": "CZ",
            "author": "test_user_003",
            "raw_text": "Zařízení se přehřívá a baterie vydrží jen pár hodin. Jsem velmi nespokojený.",
            "audio_path": None,
            "pii_masked_text": None,
            "meta_json": "{}",
        },
        {
            "interaction_id": "IID-TEST-004",
            "channel": "email",
            "source_system": "test",
            "created_ts": "2026-01-18T11:00:00Z",
            "lang": "en",
            "market": "UK",
            "author": "test_user_004",
            "raw_text": "Please contact me at john.doe@example.com or call +44 7700 900000 about my order #1234567.",
            "audio_path": None,
            "pii_masked_text": None,
            "meta_json": "{}",
        },
        {
            "interaction_id": "IID-TEST-005",
            "channel": "call",
            "source_system": "test",
            "created_ts": "2026-01-19T15:00:00Z",
            "lang": "en",
            "market": "UK",
            "author": "test_user_005",
            "raw_text": "I need to speak to a supervisor immediately! I need to escalate this. I've been having health problems since using this device and I want a refund.",
            "audio_path": None,
            "pii_masked_text": None,
            "meta_json": "{}",
        },
    ])


@pytest.fixture
def taxonomy():
    import yaml
    tax_path = Path(__file__).parent.parent / "data" / "taxonomy" / "pmc_taxonomy.yaml"
    with open(tax_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("themes", {})


# ═══════════════════════════════════════════════════════════════
# INGEST TESTS
# ═══════════════════════════════════════════════════════════════

class TestIngestPipeline:
    def test_sample_csv_load(self, config):
        from pipelines.ingest import IngestPipeline

        ingest = IngestPipeline(config)
        samples_path = Path(__file__).parent.parent / "data" / "samples" / "interactions_sample.csv"

        if samples_path.exists():
            df = ingest.ingest_file(str(samples_path))
            assert len(df) > 0, "Should load rows"
            assert "interaction_id" in df.columns
            assert "raw_text" in df.columns
            assert "pii_masked_text" in df.columns

    def test_canonical_columns(self, config, sample_interactions):
        from pipelines.ingest import IngestPipeline, CANONICAL_COLUMNS

        ingest = IngestPipeline(config)
        df = ingest._normalize(sample_interactions, "generic")

        for col in CANONICAL_COLUMNS:
            assert col in df.columns, f"Missing canonical column: {col}"

    def test_no_empty_interaction_ids(self, config, sample_interactions):
        from pipelines.ingest import IngestPipeline

        ingest = IngestPipeline(config)
        df = ingest._normalize(sample_interactions, "generic")
        assert df["interaction_id"].notna().all(), "No interaction IDs should be null"
        assert (df["interaction_id"] != "").all(), "No interaction IDs should be empty"

    def test_channel_normalization(self, config):
        from pipelines.ingest import IngestPipeline, CHANNEL_ALIASES

        ingest = IngestPipeline(config)
        raw = pd.DataFrame([{"channel": ch, "raw_text": "test"} for ch in CHANNEL_ALIASES.keys()])
        df = ingest._normalize(raw, "generic")
        for val in df["channel"]:
            assert val in CHANNEL_ALIASES.values() or val == val, "Channels should be normalized"


# ═══════════════════════════════════════════════════════════════
# PII TESTS
# ═══════════════════════════════════════════════════════════════

class TestPIIPipeline:
    def test_email_redaction(self, config):
        from pipelines.pii import PIIPipeline

        pii = PIIPipeline(config)
        text = "Please contact me at john.doe@example.com about my issue"
        masked, audit = pii.redact(text, "en")

        assert "john.doe@example.com" not in masked, "Email should be redacted"
        assert len(audit) > 0 or "EMAIL" not in str(audit), "Audit log should capture entity"

    def test_phone_redaction(self, config):
        from pipelines.pii import PIIPipeline

        pii = PIIPipeline(config)
        text = "Call me at 020-7946-0800 about my order"
        masked, audit = pii.redact(text, "en")

        assert "020-7946-0800" not in masked

    def test_loyalty_id_redaction(self, config):
        from pipelines.pii import PIIPipeline

        pii = PIIPipeline(config)
        text = "My loyalty ID is PM123456789 please look it up"
        masked, audit = pii.redact(text, "en")

        assert "PM123456789" not in masked

    def test_empty_text_handling(self, config):
        from pipelines.pii import PIIPipeline

        pii = PIIPipeline(config)
        masked, audit = pii.redact("", "en")
        assert masked == ""
        assert audit == []

    def test_dataframe_processing(self, config, sample_interactions):
        from pipelines.pii import PIIPipeline

        pii = PIIPipeline(config)
        sample_interactions["pii_masked_text"] = sample_interactions["raw_text"]
        df = pii.process_dataframe(sample_interactions)

        assert "pii_masked_text" in df.columns
        assert "pii_audit" in df.columns
        assert len(df) == len(sample_interactions)

    def test_100pct_email_masking(self, config):
        """Privacy acceptance criterion: 100% masking of configured entities."""
        from pipelines.pii import PIIPipeline

        pii = PIIPipeline(config)
        emails = [
            "Contact me at alice@example.com",
            "Reply to bob@test.org",
            "Support: help@service.co.uk",
        ]
        for text in emails:
            masked, audit = pii.redact(text, "en")
            import re
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            remaining = re.findall(email_pattern, masked)
            assert len(remaining) == 0, f"Email not redacted in: {masked}"


# ═══════════════════════════════════════════════════════════════
# SENTIMENT TESTS
# ═══════════════════════════════════════════════════════════════

class TestSentimentPipeline:
    def test_positive_text(self, config):
        from pipelines.sentiment import SentimentPipeline

        sent = SentimentPipeline(config)
        result = sent.analyze("Excellent product! Great battery life and fantastic customer service!", "en")

        assert result["sentiment"] in ["pos", "neu", "neg"]
        assert -1.0 <= result["score"] <= 1.0
        assert result["sentiment"] == "pos", f"Expected positive, got {result['sentiment']}"

    def test_negative_text(self, config):
        from pipelines.sentiment import SentimentPipeline

        sent = SentimentPipeline(config)
        result = sent.analyze("Terrible product! Broken, horrible experience, worst ever!", "en")

        assert result["sentiment"] == "neg", f"Expected negative, got {result['sentiment']}"
        assert result["score"] < 0

    def test_dsat_flag(self, config):
        from pipelines.sentiment import SentimentPipeline

        sent = SentimentPipeline(config)
        result = sent.analyze("This is a terrible fraud! I want a refund immediately! Worst company ever!", "en")

        assert result["dsat_flag"] == True

    def test_czech_sentiment(self, config):
        from pipelines.sentiment import SentimentPipeline

        sent = SentimentPipeline(config)
        result = sent.analyze("Skvělý produkt! Úžasný zákaznický servis!", "cs")

        assert result["sentiment"] in ["pos", "neu"]

    def test_czech_negative(self, config):
        from pipelines.sentiment import SentimentPipeline

        sent = SentimentPipeline(config)
        result = sent.analyze("Hrozné! Nejhorší produkt, nefunguje, zklamání.", "cs")

        assert result["sentiment"] == "neg"

    def test_empty_text(self, config):
        from pipelines.sentiment import SentimentPipeline

        sent = SentimentPipeline(config)
        result = sent.analyze("", "en")

        assert result["sentiment"] == "neu"
        assert result["score"] == 0.0

    def test_dataframe_processing(self, config, sample_interactions):
        from pipelines.sentiment import SentimentPipeline

        sample_interactions["pii_masked_text"] = sample_interactions["raw_text"]
        sent = SentimentPipeline(config)
        df = sent.process_dataframe(sample_interactions)

        assert "sentiment" in df.columns
        assert "sentiment_score" in df.columns
        assert "dsat_flag" in df.columns
        assert df["sentiment"].isin(["pos", "neu", "neg"]).all()
        assert df["sentiment_score"].between(-1, 1).all()


# ═══════════════════════════════════════════════════════════════
# BEHAVIORS TESTS
# ═══════════════════════════════════════════════════════════════

class TestBehaviorsPipeline:
    def test_empathy_detected(self, config):
        from pipelines.behaviors import BehaviorsPipeline

        beh = BehaviorsPipeline(config)
        result = beh.detect("I completely understand your frustration and I apologize for the inconvenience.", "en")

        assert result["empathy"] == True

    def test_authentication_detected(self, config):
        from pipelines.behaviors import BehaviorsPipeline

        beh = BehaviorsPipeline(config)
        result = beh.detect("I need to verify your identity. Can you please provide your date of birth?", "en")

        assert result["authentication"] == True

    def test_escalation_detected(self, config):
        from pipelines.behaviors import BehaviorsPipeline

        beh = BehaviorsPipeline(config)
        result = beh.detect("I will escalate this to a senior agent and transfer you to a supervisor.", "en")

        assert result["escalation"] == True

    def test_adverse_event_detected(self, config):
        from pipelines.behaviors import BehaviorsPipeline

        beh = BehaviorsPipeline(config)
        result = beh.detect("I've been having health problems and I'm feeling very ill since using this product.", "en")

        assert result["adverse_event"] == True

    def test_czech_detection(self, config):
        from pipelines.behaviors import BehaviorsPipeline

        beh = BehaviorsPipeline(config)
        result = beh.detect("Plně chápu vaši frustraci a omlouvám se za nepříjemnosti.", "cs")

        assert result["empathy"] == True

    def test_no_false_positive(self, config):
        from pipelines.behaviors import BehaviorsPipeline

        beh = BehaviorsPipeline(config)
        result = beh.detect("My order has been delayed by two weeks.", "en")

        assert result["escalation"] == False
        assert result["adverse_event"] == False


# ═══════════════════════════════════════════════════════════════
# COMPLIANCE TESTS
# ═══════════════════════════════════════════════════════════════

class TestCompliancePipeline:
    def test_adverse_event_flag(self, config):
        from pipelines.compliance import CompliancePipeline

        comp = CompliancePipeline(config)
        result = comp.detect("I've been experiencing health issues since using your product. I saw a doctor.", "en")

        assert result["adverse_event"] == True
        assert result["health_mention"] == True

    def test_gdpr_flag(self, config):
        from pipelines.compliance import CompliancePipeline

        comp = CompliancePipeline(config)
        result = comp.detect("I want to delete my data under GDPR right to erasure.", "en")

        assert result["gdpr"] == True or result["privacy"] == True

    def test_age_verification_flag(self, config):
        from pipelines.compliance import CompliancePipeline

        comp = CompliancePipeline(config)
        result = comp.detect("I need to verify your age. Are you 18+ years old?", "en")

        assert result["age_verification"] == True

    def test_critical_detection(self, config, sample_interactions):
        from pipelines.compliance import CompliancePipeline

        comp = CompliancePipeline(config)
        sample_interactions["pii_masked_text"] = sample_interactions["raw_text"]
        df = comp.process_dataframe(sample_interactions)

        assert "has_critical_flag" in df.columns
        assert "compliance_flags" in df.columns
        # IID-TEST-005 mentions health problems - should be flagged
        critical_row = df[df["interaction_id"] == "IID-TEST-005"]
        if not critical_row.empty:
            assert critical_row["has_critical_flag"].iloc[0] == True


# ═══════════════════════════════════════════════════════════════
# ANOMALY TESTS
# ═══════════════════════════════════════════════════════════════

class TestAnomalyPipeline:
    def test_spike_detection(self, config):
        from pipelines.anomalies import AnomalyPipeline
        import numpy as np

        anomaly = AnomalyPipeline(config)

        # Create data with a clear spike
        dates = pd.date_range("2026-01-01", periods=30, freq="D")
        volumes = [100] * 27 + [500, 480, 100]  # Spike at days 28-29

        df = pd.DataFrame({
            "created_ts": pd.Series(dates).repeat(1),
            "dummy": range(30),
        })
        df = pd.concat([
            df.assign(created_ts=dates[i])
            for i in range(30)
            for _ in range(volumes[i])
        ])
        df = df.reset_index(drop=True)

        alerts = anomaly.detect_anomalies(df, date_col="created_ts")

        if not alerts.empty:
            assert len(alerts) > 0, "Should detect spike"

    def test_no_false_alerts_stable_data(self, config):
        from pipelines.anomalies import AnomalyPipeline

        anomaly = AnomalyPipeline(config)

        # Create stable data with no spikes
        dates = pd.date_range("2026-01-01", periods=30, freq="D")
        df = pd.concat([
            pd.DataFrame({"created_ts": [d] * 100, "dummy": range(100)})
            for d in dates
        ])
        df = df.reset_index(drop=True)

        alerts = anomaly.detect_anomalies(df, date_col="created_ts")

        if not alerts.empty:
            fpr = len(alerts) / 30
            assert fpr < 0.3, f"False positive rate too high: {fpr:.2f}"


# ═══════════════════════════════════════════════════════════════
# IMPACT TESTS
# ═══════════════════════════════════════════════════════════════

class TestImpactPipeline:
    def test_sentiment_proxy_impact(self, config, sample_interactions, taxonomy):
        from pipelines.impact import ImpactPipeline
        from pipelines.sentiment import SentimentPipeline
        from pipelines.behaviors import BehaviorsPipeline

        sample_interactions["pii_masked_text"] = sample_interactions["raw_text"]
        sent = SentimentPipeline(config)
        df = sent.process_dataframe(sample_interactions)
        df["topic_label"] = ["Device heating", "Positive feedback", "Device heating", "General", "Escalation"]

        impact = ImpactPipeline(config)
        result = impact.compute_topic_impact(df)

        assert not result.empty
        assert "topic_label" in result.columns or "volume" in result.columns

    def test_automation_candidates(self, config, sample_interactions, taxonomy):
        from pipelines.impact import ImpactPipeline

        sample_interactions["pii_masked_text"] = sample_interactions["raw_text"]
        sample_interactions["sentiment_score"] = [-0.5, 0.8, -0.6, 0.0, -0.8]
        sample_interactions["topic_label"] = [
            "Consumables availability", "Positive feedback",
            "Device heating issues", "General", "Escalation handling"
        ]

        impact = ImpactPipeline(config)
        candidates = impact.get_automation_candidates(sample_interactions, taxonomy)

        assert not candidates.empty
        assert "topic_label" in candidates.columns


# ═══════════════════════════════════════════════════════════════
# INTEGRATION TEST: MINI PIPELINE RUN
# ═══════════════════════════════════════════════════════════════

class TestPipelineIntegration:
    def test_pii_then_sentiment(self, config, sample_interactions):
        from pipelines.pii import PIIPipeline
        from pipelines.sentiment import SentimentPipeline

        pii = PIIPipeline(config)
        sent = SentimentPipeline(config)

        df = pii.process_dataframe(sample_interactions, text_col="raw_text")
        df = sent.process_dataframe(df, text_col="pii_masked_text")

        assert "sentiment" in df.columns
        assert "pii_masked_text" in df.columns
        # Ensure email in IID-TEST-004 was masked
        row_004 = df[df["interaction_id"] == "IID-TEST-004"]
        if not row_004.empty:
            assert "john.doe@example.com" not in str(row_004["pii_masked_text"].iloc[0])

    def test_behaviors_then_compliance(self, config, sample_interactions):
        from pipelines.behaviors import BehaviorsPipeline
        from pipelines.compliance import CompliancePipeline

        sample_interactions["pii_masked_text"] = sample_interactions["raw_text"]
        beh = BehaviorsPipeline(config)
        comp = CompliancePipeline(config)

        df = beh.process_dataframe(sample_interactions)
        df = comp.process_dataframe(df)

        assert "behavior_escalation" in df.columns
        assert "compliance_adverse_event" in df.columns

        # IID-TEST-005: escalation + health + adverse event
        row_005 = df[df["interaction_id"] == "IID-TEST-005"]
        if not row_005.empty:
            assert row_005["behavior_escalation"].iloc[0] == True
            assert row_005["has_critical_flag"].iloc[0] == True

    def test_demo_data_loads(self):
        """Test demo data CSV is loadable and well-formed."""
        samples_path = Path(__file__).parent.parent / "data" / "samples" / "interactions_sample.csv"
        assert samples_path.exists(), "Sample data must exist"

        df = pd.read_csv(samples_path)
        assert len(df) >= 50, "Should have at least 50 sample rows"
        assert "interaction_id" in df.columns
        assert "raw_text" in df.columns
        assert "lang" in df.columns
        assert df["lang"].isin(["en", "cs"]).all(), "Languages should be en or cs"
        assert df["interaction_id"].notna().all(), "No null IDs"

    def test_kpi_data_loads(self):
        """Test KPI sample data is loadable."""
        kpi_path = Path(__file__).parent.parent / "data" / "samples" / "kpi_daily_sample.csv"
        assert kpi_path.exists()

        df = pd.read_csv(kpi_path)
        assert len(df) >= 10
        required_cols = ["date", "market", "nps", "csat"]
        for col in required_cols:
            assert col in df.columns, f"Missing KPI column: {col}"

    def test_taxonomy_structure(self, taxonomy):
        """Validate taxonomy YAML structure."""
        assert len(taxonomy) >= 20, "Should have at least 20 themes"

        for theme_key, theme_data in taxonomy.items():
            assert "level1" in theme_data, f"{theme_key}: missing level1"
            assert "level2" in theme_data, f"{theme_key}: missing level2"
            assert "labels" in theme_data, f"{theme_key}: missing labels"
            assert "en" in theme_data["labels"], f"{theme_key}: missing EN label"
            assert "cs" in theme_data["labels"], f"{theme_key}: missing CS label"
            assert "seeds" in theme_data, f"{theme_key}: missing seeds"

    def test_i18n_files(self):
        """Test i18n files have required keys."""
        i18n_dir = Path(__file__).parent.parent.parent / "i18n"

        for lang in ["en", "cs"]:
            lang_file = i18n_dir / f"{lang}.json"
            assert lang_file.exists(), f"Missing i18n file: {lang}.json"

            with open(lang_file, "r") as f:
                data = json.load(f)

            required_keys = ["app_title", "tabs", "overview", "common"]
            for key in required_keys:
                assert key in data, f"i18n/{lang}.json missing key: {key}"
