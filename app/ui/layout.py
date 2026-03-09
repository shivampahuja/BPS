"""
Page layout definitions and helper functions for the CII Gradio app.
"""
import json
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure app root is in path
APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(APP_DIR))

import pandas as pd
import gradio as gr

from ui.components import (
    kpi_row_html,
    alerts_panel_html,
    pipeline_status_html,
    taxonomy_display_html,
    sentiment_donut,
    topic_bar_chart,
    sentiment_trend_chart,
    volume_trend_chart,
    behavior_heatmap,
    nps_topic_scatter,
    compliance_gauge,
    impact_waterfall,
    empty_chart,
    COLORS,
)


def get_i18n(lang: str = "en") -> dict:
    """Load i18n strings for given language."""
    i18n_dir = Path(__file__).parent.parent.parent / "i18n"
    lang_file = i18n_dir / f"{lang}.json"
    fallback_file = i18n_dir / "en.json"

    target = lang_file if lang_file.exists() else fallback_file
    try:
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def format_kpi_value(val, fmt: str = "number", prefix: str = "", suffix: str = "") -> str:
    """Format a KPI value for display."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    try:
        if fmt == "percent":
            return f"{prefix}{float(val):.1f}%{suffix}"
        elif fmt == "integer":
            return f"{prefix}{int(val):,}{suffix}"
        elif fmt == "score":
            return f"{prefix}{float(val):.2f}{suffix}"
        else:
            return f"{prefix}{float(val):.1f}{suffix}"
    except Exception:
        return str(val)


def build_overview_kpis(state: dict, lang: str = "en") -> str:
    """Build KPI tiles HTML for overview page."""
    i18n = get_i18n(lang)
    t = i18n.get("overview", {})

    df = state.get("interactions", pd.DataFrame())
    kpi_df = state.get("kpi_data", pd.DataFrame())

    total = len(df) if not df.empty else 0
    avg_sentiment = float(df["sentiment_score"].mean()) if not df.empty and "sentiment_score" in df.columns else 0.0
    dsat_rate = float(df["dsat_flag"].mean()) * 100 if not df.empty and "dsat_flag" in df.columns else 0.0

    nps = float(kpi_df["nps"].mean()) if not kpi_df.empty and "nps" in kpi_df.columns else 0.0
    csat = float(kpi_df["csat"].mean()) if not kpi_df.empty and "csat" in kpi_df.columns else 0.0
    aht = float(kpi_df["aht_sec"].mean()) if not kpi_df.empty and "aht_sec" in kpi_df.columns else 0.0

    kpis = [
        {
            "label": t.get("kpi_volume", "Total Interactions"),
            "value": f"{total:,}",
            "icon": "📊",
            "color": COLORS["primary"],
        },
        {
            "label": t.get("kpi_nps", "NPS Score"),
            "value": format_kpi_value(nps, "score"),
            "icon": "⭐",
            "color": COLORS["positive"] if nps >= 30 else COLORS["warning"] if nps >= 0 else COLORS["negative"],
        },
        {
            "label": t.get("kpi_csat", "CSAT"),
            "value": format_kpi_value(csat, "score"),
            "icon": "😊",
            "color": COLORS["positive"] if csat >= 4 else COLORS["warning"],
        },
        {
            "label": t.get("kpi_sentiment", "Avg Sentiment"),
            "value": format_kpi_value(avg_sentiment, "score"),
            "icon": "💬",
            "color": COLORS["positive"] if avg_sentiment > 0.1 else COLORS["negative"] if avg_sentiment < -0.1 else COLORS["neutral"],
        },
        {
            "label": t.get("kpi_dsat_rate", "DSAT Rate"),
            "value": format_kpi_value(dsat_rate, "percent"),
            "icon": "⚡",
            "color": COLORS["negative"] if dsat_rate > 30 else COLORS["warning"] if dsat_rate > 15 else COLORS["positive"],
        },
        {
            "label": t.get("kpi_aht", "Avg Handle Time"),
            "value": format_kpi_value(aht / 60, "score", suffix=" min") if aht > 0 else "N/A",
            "icon": "⏱️",
            "color": COLORS["primary"],
        },
    ]

    return kpi_row_html(kpis)


def build_topic_summary_df(state: dict, lang: str = "en") -> pd.DataFrame:
    """Build topic summary table for display."""
    df = state.get("interactions", pd.DataFrame())
    if df.empty or "topic_label" not in df.columns:
        return pd.DataFrame()

    agg = {}
    agg["volume"] = ("topic_label", "count")

    if "sentiment_score" in df.columns:
        agg["avg_sentiment"] = ("sentiment_score", "mean")

    if "dsat_flag" in df.columns:
        agg["dsat_rate"] = ("dsat_flag", "mean")

    topic_stats = df.groupby("topic_label").agg(**agg).reset_index()

    if "avg_sentiment" in topic_stats.columns:
        topic_stats["avg_sentiment"] = topic_stats["avg_sentiment"].round(3)

    if "dsat_rate" in topic_stats.columns:
        topic_stats["dsat_rate"] = (topic_stats["dsat_rate"] * 100).round(1)

    if "l1_domain" in df.columns:
        domain_map = df.groupby("topic_label")["l1_domain"].first()
        topic_stats["domain"] = topic_stats["topic_label"].map(domain_map)

    return topic_stats.sort_values("volume", ascending=False)


def export_to_csv(df: pd.DataFrame, filename: str = "export.csv") -> str:
    """Export DataFrame to CSV file and return path."""
    exports_dir = Path(__file__).parent.parent / "exports"
    exports_dir.mkdir(exist_ok=True)
    path = exports_dir / filename
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def export_to_json(df: pd.DataFrame, filename: str = "export.json") -> str:
    """Export DataFrame to JSON file and return path."""
    exports_dir = Path(__file__).parent.parent / "exports"
    exports_dir.mkdir(exist_ok=True)
    path = exports_dir / filename
    df.to_json(path, orient="records", indent=2, default_handler=str)
    return str(path)


def get_capa_list(state: dict) -> pd.DataFrame:
    """Generate CAPA list from alerts and DSAT interactions."""
    rows = []

    alerts = state.get("alerts", pd.DataFrame())
    if not alerts.empty:
        for _, alert in alerts.iterrows():
            rows.append({
                "capa_type": "alert",
                "priority": "high" if alert.get("z_score", 0) > 3 else "medium",
                "description": alert.get("explanation", ""),
                "metric": alert.get("metric", ""),
                "date": str(alert.get("date", ""))[:10],
                "recommended_action": alert.get("playbook_en", ""),
                "status": "open",
            })

    df = state.get("interactions", pd.DataFrame())
    if not df.empty and "compliance_flags" in df.columns:
        critical = df[df.get("has_critical_flag", pd.Series(False, index=df.index))]
        for _, row in critical.head(20).iterrows():
            rows.append({
                "capa_type": "compliance",
                "priority": "critical",
                "description": f"Critical compliance flag: {row.get('critical_flags', [])}",
                "metric": "compliance",
                "date": str(row.get("created_ts", ""))[:10],
                "recommended_action": "Immediate review and routing per SOP",
                "status": "open",
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def get_automation_backlog(state: dict) -> pd.DataFrame:
    """Get automation candidate backlog for export."""
    df = state.get("automation_candidates", pd.DataFrame())
    if df.empty:
        return pd.DataFrame()

    result = df[df.get("automation_eligible", pd.Series(False, index=df.index)) == True].copy() if "automation_eligible" in df.columns else df
    result["backlog_type"] = "FAQ/Bot/VCA"
    result["status"] = "pending"
    return result


def sample_verbatims(
    df: pd.DataFrame,
    topic: Optional[str] = None,
    sentiment: Optional[str] = None,
    n: int = 10,
    lang: Optional[str] = None,
) -> pd.DataFrame:
    """Get sample verbatim interactions with optional filters."""
    if df.empty:
        return pd.DataFrame()

    filtered = df.copy()

    if topic and topic != "All" and "topic_label" in filtered.columns:
        filtered = filtered[filtered["topic_label"].str.contains(topic, case=False, na=False)]

    if sentiment and sentiment != "All" and "sentiment" in filtered.columns:
        filtered = filtered[filtered["sentiment"] == sentiment]

    if lang and lang != "All" and "lang" in filtered.columns:
        filtered = filtered[filtered["lang"] == lang]

    display_cols = []
    for col in ["interaction_id", "created_ts", "channel", "lang", "market",
                "pii_masked_text", "sentiment", "sentiment_score",
                "topic_label", "behaviors", "compliance_flags"]:
        if col in filtered.columns:
            display_cols.append(col)

    return filtered[display_cols].head(n)
