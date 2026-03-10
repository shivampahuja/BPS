"""
CII Web Application — FastAPI backend
Serves the SPA dashboard + all REST API endpoints.
"""
import json
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

# ── Path ─────────────────────────────────────────────────────
APP_DIR = Path(__file__).parent
sys.path.insert(0, str(APP_DIR))

import pandas as pd
import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ── Pipeline imports ──────────────────────────────────────────
from pipelines.ingest import IngestPipeline
from pipelines.pii import PIIPipeline
from pipelines.sentiment import SentimentPipeline
from pipelines.behaviors import BehaviorsPipeline
from pipelines.compliance import CompliancePipeline
from pipelines.anomalies import AnomalyPipeline
from pipelines.impact import ImpactPipeline
from pipelines.embed import EmbedPipeline
from pipelines.rag import RAGPipeline

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cii.webapp")

# ── Config & taxonomy ─────────────────────────────────────────
with open(APP_DIR / "config.yaml") as f:
    CONFIG = yaml.safe_load(f)

with open(APP_DIR / "data" / "taxonomy" / "pmc_taxonomy.yaml") as f:
    TAX_DATA = yaml.safe_load(f)
TAXONOMY = TAX_DATA.get("themes", {})

SAMPLES_DIR = APP_DIR / "data" / "samples"

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(title="CII Platform", version="1.0.0", docs_url="/api/docs")

app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# ── In-memory store (demo) ────────────────────────────────────
_store: dict = {
    "interactions": None,
    "kpi_data": None,
    "alerts": None,
    "automation": None,
}

# ── Pipeline singletons ───────────────────────────────────────
_pipes: dict = {}

def pipe(name: str):
    if name not in _pipes:
        if name == "ingest":    _pipes[name] = IngestPipeline(CONFIG)
        elif name == "pii":     _pipes[name] = PIIPipeline(CONFIG)
        elif name == "sentiment": _pipes[name] = SentimentPipeline(CONFIG)
        elif name == "behaviors": _pipes[name] = BehaviorsPipeline(CONFIG)
        elif name == "compliance": _pipes[name] = CompliancePipeline(CONFIG)
        elif name == "anomalies": _pipes[name] = AnomalyPipeline(CONFIG)
        elif name == "impact":  _pipes[name] = ImpactPipeline(CONFIG)
        elif name == "embed":   _pipes[name] = EmbedPipeline(CONFIG)
        elif name == "rag":
            _pipes[name] = RAGPipeline(CONFIG, pipe("embed"))
    return _pipes[name]

# ── Demo data loader (reuse from app.py logic) ────────────────
def _mock_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Attach lightweight mock analysis columns for demo speed."""
    import re, json as _json

    neg_words = ["frustrated","terrible","broken","problem","unacceptable","worst",
                 "frustrating","delay","damaged","fraud","incorrect","overheating",
                 "frustrující","hrozné","rozbité","problém","zpožděné","poškozené"]
    pos_words = ["excellent","great","fantastic","amazing","perfect","brilliant","wonderful",
                 "skvělý","výborný","úžasný","fantastický","perfektní","happy","resolved"]

    def score(t):
        t = str(t).lower()
        n = sum(1 for w in neg_words if w in t)
        p = sum(1 for w in pos_words if w in t)
        if n > p: return max(-0.9, -0.3*n), "neg"
        if p > n: return min(0.9, 0.3*p), "pos"
        return 0.0, "neu"

    sv = df["raw_text"].apply(score)
    df["sentiment_score"] = [x[0] for x in sv]
    df["sentiment"] = [x[1] for x in sv]
    df["dsat_flag"] = df["sentiment_score"] < -0.3

    def parse_meta(m):
        try: return _json.loads(str(m)) if m and str(m) != "nan" else {}
        except: return {}

    metas = df["meta_json"].apply(parse_meta)

    TOPIC_MAP = {
        "Device Heating": "Device heating issues", "Battery Charge": "Battery & charging issues",
        "Device Breakage": "Device breakage & durability", "Consumables Availability": "Consumables availability",
        "Oos Stock": "Out of stock & inventory", "Flavor Quality": "Flavor & quality issues",
        "Replacement Warranty": "Replacement & warranty process", "Agent Empathy": "Agent empathy & active listening",
        "Fcr Resolution": "First contact resolution", "Escalation": "Escalation handling",
        "Authentication": "Authentication & identity verification", "Adverse Event": "Adverse event & health mention",
        "Checkout Friction": "Checkout friction", "Search Navigation": "Search & navigation issues",
        "Login Account": "Login & account access issues", "Bot Faq": "Bot & FAQ quality",
        "Delivery Delay": "Delivery delays", "Damaged Delivery": "Damaged in transit",
        "Payment Refund": "Payment & refund issues", "Order Management": "Order amendments & cancellations",
        "Promo Pricing": "Promo & discount issues", "Onboarding": "Onboarding & first use guidance",
        "Rewards Loyalty": "Rewards & loyalty points", "Subscription": "Subscription & trade-in",
        "Instore Service": "In-store service quality", "Privacy Concern": "Privacy & data concerns",
        "Age Verification": "Age & nicotine verification",
    }

    DOMAIN_MAP = {
        "Device heating issues": "Product Experience", "Battery & charging issues": "Product Experience",
        "Device breakage & durability": "Product Experience", "Consumables availability": "Product Experience",
        "Flavor & quality issues": "Product Experience", "Replacement & warranty process": "Product Experience",
        "Agent empathy & active listening": "Service & Support (CSC)", "First contact resolution": "Service & Support (CSC)",
        "Escalation handling": "Service & Support (CSC)", "Authentication & identity verification": "Service & Support (CSC)",
        "Adverse event & health mention": "Service & Support (CSC)", "Proactive outreach & recovery": "Service & Support (CSC)",
        "Checkout friction": "Digital Experience", "Search & navigation issues": "Digital Experience",
        "Login & account access issues": "Digital Experience", "Bot & FAQ quality": "Digital Experience",
        "Promo & discount issues": "Digital Experience", "Delivery delays": "Commerce & Fulfillment",
        "Damaged in transit": "Commerce & Fulfillment", "Out of stock & inventory": "Commerce & Fulfillment",
        "Payment & refund issues": "Commerce & Fulfillment", "Order amendments & cancellations": "Commerce & Fulfillment",
        "Onboarding & first use guidance": "Program & Loyalty", "Rewards & loyalty points": "Program & Loyalty",
        "Subscription & trade-in": "Program & Loyalty", "In-store service quality": "Channel Experience",
        "Privacy & data concerns": "Policy & Compliance", "Age & nicotine verification": "Policy & Compliance",
    }

    def get_topic(meta):
        topics = meta.get("topics", [])
        if topics:
            raw = topics[0].replace("THEME_", "").replace("_", " ").title()
            return TOPIC_MAP.get(raw, raw)
        return "General Inquiry"

    df["topic_label"] = metas.apply(get_topic)
    df["l1_domain"] = df["topic_label"].map(DOMAIN_MAP).fillna("Service & Support (CSC)")

    def beh(row):
        t = str(row.get("raw_text","")).lower()
        b = []
        if any(w in t for w in ["understand","apolog","sorry","chápu","omlouvám"]): b.append("empathy")
        if any(w in t for w in ["verify","authentication","date of birth","ověřit"]): b.append("authentication")
        if any(w in t for w in ["resolved","fixed","anything else","vyřešeno"]): b.append("resolution_confirmation")
        if any(w in t for w in ["escalate","supervisor","manager","eskalovat"]): b.append("escalation")
        if any(w in t for w in ["health","doctor","hospital","zdraví","lékař"]): b.append("adverse_event")
        return b

    df["behaviors"] = df.apply(beh, axis=1)
    for b in ["empathy","authentication","resolution_confirmation","escalation","adverse_event"]:
        df[f"behavior_{b}"] = df["behaviors"].apply(lambda x: b in x)

    def comp(row):
        t = str(row.get("raw_text","")).lower()
        flags = []
        if any(w in t for w in ["health","doctor","medical","adverse","zdraví","lékař"]): flags += ["adverse_event","health_mention"]
        if any(w in t for w in ["gdpr","privacy","delete my data","soukromí"]): flags += ["privacy","gdpr"]
        if any(w in t for w in ["age verification","underage","ověření věku"]): flags += ["age_verification"]
        if any(w in t for w in ["nicotine","nikotin"]): flags += ["nicotine_declaration"]
        return list(set(flags))

    df["compliance_flags"] = df.apply(comp, axis=1)
    df["critical_flags"] = df["compliance_flags"].apply(lambda x: [f for f in x if f in ["adverse_event","health_mention"]])
    df["has_critical_flag"] = df["critical_flags"].apply(bool)
    df["has_compliance_flag"] = df["compliance_flags"].apply(bool)
    df["queue"] = metas.apply(lambda m: m.get("queue","tier1"))
    df["vendor"] = metas.apply(lambda m: m.get("vendor","vendorA"))
    df["nps_score"] = metas.apply(lambda m: m.get("nps_score", None))
    return df


def _ensure_demo():
    """Load demo data into _store if not already loaded."""
    if _store["interactions"] is not None:
        return

    df = pd.read_csv(SAMPLES_DIR / "interactions_sample.csv")
    df["created_ts"] = pd.to_datetime(df["created_ts"], errors="coerce")
    df = _mock_analysis(df)
    _store["interactions"] = df

    kpi = pd.read_csv(SAMPLES_DIR / "kpi_daily_sample.csv")
    kpi["date"] = pd.to_datetime(kpi["date"], errors="coerce")
    _store["kpi_data"] = kpi

    with open(SAMPLES_DIR / "alerts_sample.json") as f:
        alerts = pd.DataFrame(json.load(f))
    _store["alerts"] = alerts

    auto = pipe("impact").get_automation_candidates(df, TAXONOMY)
    _store["automation"] = auto
    logger.info("Demo data loaded")


# ═══════════════════════════════════════════════════════════════
# ROUTES — HTML
# ═══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ═══════════════════════════════════════════════════════════════
# API — OVERVIEW
# ═══════════════════════════════════════════════════════════════

@app.get("/api/overview")
async def api_overview():
    _ensure_demo()
    df = _store["interactions"]
    kpi = _store["kpi_data"]
    alerts = _store["alerts"]

    total = len(df)
    avg_sent = float(df["sentiment_score"].mean())
    dsat_rate = float(df["dsat_flag"].mean()) * 100
    neg_pct = float((df["sentiment"] == "neg").mean()) * 100
    pos_pct = float((df["sentiment"] == "pos").mean()) * 100
    neu_pct = float((df["sentiment"] == "neu").mean()) * 100
    nps = float(kpi["nps"].mean()) if "nps" in kpi.columns else 0
    csat = float(kpi["csat"].mean()) if "csat" in kpi.columns else 0
    aht = float(kpi["aht_sec"].mean()) / 60 if "aht_sec" in kpi.columns else 0

    # Volume trend (daily)
    df2 = df.copy()
    df2["date"] = df2["created_ts"].dt.date.astype(str)
    vol_trend = df2.groupby("date").size().reset_index(name="count")
    vol_trend = vol_trend.sort_values("date").tail(30)

    # Sentiment trend
    sent_trend = df2.groupby("date")["sentiment_score"].mean().reset_index()
    sent_trend.columns = ["date", "avg"]
    sent_trend = sent_trend.sort_values("date").tail(30)

    # Top topics
    top_topics = (
        df.groupby("topic_label")
        .agg(volume=("topic_label","count"), avg_sentiment=("sentiment_score","mean"))
        .reset_index()
        .sort_values("volume", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

    # Channel breakdown
    channels = df["channel"].value_counts().reset_index()
    channels.columns = ["channel","count"]

    # Market breakdown
    markets = df["market"].value_counts().reset_index()
    markets.columns = ["market","count"]

    return {
        "kpis": {
            "total_interactions": total,
            "nps": round(nps, 1),
            "csat": round(csat, 2),
            "avg_sentiment": round(avg_sent, 3),
            "dsat_rate": round(dsat_rate, 1),
            "aht_min": round(aht, 1),
            "pos_pct": round(pos_pct, 1),
            "neg_pct": round(neg_pct, 1),
            "neu_pct": round(neu_pct, 1),
        },
        "volume_trend": vol_trend.to_dict(orient="records"),
        "sentiment_trend": sent_trend.to_dict(orient="records"),
        "top_topics": top_topics,
        "channels": channels.to_dict(orient="records"),
        "markets": markets.to_dict(orient="records"),
        "alerts": alerts.head(5).to_dict(orient="records") if alerts is not None else [],
        "critical_count": int(df["has_critical_flag"].sum()) if "has_critical_flag" in df.columns else 0,
        "compliance_count": int(df["has_compliance_flag"].sum()) if "has_compliance_flag" in df.columns else 0,
    }


# ═══════════════════════════════════════════════════════════════
# API — TOPICS / THEMES
# ═══════════════════════════════════════════════════════════════

@app.get("/api/topics")
async def api_topics(domain: str = "All", market: str = "All", lang: str = "All", channel: str = "All"):
    _ensure_demo()
    df = _store["interactions"].copy()

    if domain != "All" and "l1_domain" in df.columns: df = df[df["l1_domain"] == domain]
    if market != "All": df = df[df["market"] == market]
    if lang != "All": df = df[df["lang"] == lang]
    if channel != "All": df = df[df["channel"] == channel]

    if df.empty:
        return {"topics": [], "scatter": []}

    topic_stats = (
        df.groupby(["topic_label","l1_domain"])
        .agg(
            volume=("topic_label","count"),
            avg_sentiment=("sentiment_score","mean"),
            dsat_rate=("dsat_flag","mean"),
        )
        .reset_index()
    )
    topic_stats["avg_sentiment"] = topic_stats["avg_sentiment"].round(3)
    topic_stats["dsat_pct"] = (topic_stats["dsat_rate"] * 100).round(1)
    topic_stats = topic_stats.sort_values("volume", ascending=False)

    # Behavior overlay
    beh_cols = [c for c in df.columns if c.startswith("behavior_")]
    if beh_cols:
        beh_agg = df.groupby("topic_label")[beh_cols].mean().mul(100).round(1)
        beh_agg.columns = [c.replace("behavior_","") for c in beh_agg.columns]
        beh_dict = beh_agg.to_dict(orient="index")
    else:
        beh_dict = {}

    # Compliance overlay
    comp_cols = [c for c in df.columns if c.startswith("compliance_")]
    if comp_cols:
        comp_agg = df.groupby("topic_label")[comp_cols].sum()
        comp_agg.columns = [c.replace("compliance_","") for c in comp_agg.columns]
        comp_dict = comp_agg.to_dict(orient="index")
    else:
        comp_dict = {}

    records = topic_stats.to_dict(orient="records")
    for r in records:
        r["behaviors"] = beh_dict.get(r["topic_label"], {})
        r["compliance"] = comp_dict.get(r["topic_label"], {})

    return {"topics": records}


@app.get("/api/topics/{topic_label}/samples")
async def api_topic_samples(topic_label: str, n: int = 10):
    _ensure_demo()
    df = _store["interactions"]
    subset = df[df["topic_label"] == topic_label]
    cols = [c for c in ["interaction_id","created_ts","channel","lang","market",
                        "pii_masked_text","sentiment","sentiment_score","behaviors","compliance_flags"]
            if c in subset.columns]
    return {"samples": subset[cols].head(n).to_dict(orient="records")}


# ═══════════════════════════════════════════════════════════════
# API — SENTIMENT
# ═══════════════════════════════════════════════════════════════

@app.get("/api/sentiment")
async def api_sentiment(market: str = "All", channel: str = "All", lang: str = "All"):
    _ensure_demo()
    df = _store["interactions"].copy()
    if market != "All": df = df[df["market"] == market]
    if channel != "All": df = df[df["channel"] == channel]
    if lang != "All": df = df[df["lang"] == lang]

    dist = {
        "pos": int((df["sentiment"]=="pos").sum()),
        "neu": int((df["sentiment"]=="neu").sum()),
        "neg": int((df["sentiment"]=="neg").sum()),
        "dsat": int(df["dsat_flag"].sum()),
        "total": len(df),
    }

    by_channel = df.groupby("channel")["sentiment_score"].mean().round(3).to_dict()
    by_market = df.groupby("market")["sentiment_score"].mean().round(3).to_dict()

    dsat_topics = (
        df[df["dsat_flag"]]
        .groupby("topic_label").size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

    negatives = df[df["sentiment"]=="neg"].sort_values("sentiment_score").head(15)
    cols = [c for c in ["interaction_id","created_ts","channel","lang","market",
                        "pii_masked_text","sentiment_score","topic_label"] if c in negatives.columns]
    negatives_list = negatives[cols].to_dict(orient="records")

    return {
        "distribution": dist,
        "by_channel": [{"channel":k,"avg_sentiment":v} for k,v in by_channel.items()],
        "by_market": [{"market":k,"avg_sentiment":v} for k,v in by_market.items()],
        "dsat_topics": dsat_topics,
        "negatives": negatives_list,
    }


# ═══════════════════════════════════════════════════════════════
# API — OPERATIONS
# ═══════════════════════════════════════════════════════════════

@app.get("/api/operations")
async def api_operations(group_by: str = "channel"):
    _ensure_demo()
    df = _store["interactions"]

    beh_cols = [c for c in df.columns if c.startswith("behavior_")]
    overall_beh = {c.replace("behavior_",""): round(float(df[c].mean())*100,1) for c in beh_cols}

    comp_cols = [c for c in df.columns if c.startswith("compliance_")]
    overall_comp = {c.replace("compliance_",""): int(df[c].sum()) for c in comp_cols}

    if group_by in df.columns and beh_cols:
        heatmap_data = []
        for grp_val in df[group_by].dropna().unique():
            grp_df = df[df[group_by]==grp_val]
            row = {group_by: grp_val}
            for c in beh_cols:
                row[c.replace("behavior_","")] = round(float(grp_df[c].mean())*100,1)
            heatmap_data.append(row)
    else:
        heatmap_data = []

    critical_df = df[df["has_critical_flag"]==True] if "has_critical_flag" in df.columns else pd.DataFrame()
    cols = [c for c in ["interaction_id","created_ts","channel","market","pii_masked_text","critical_flags"] if c in critical_df.columns]
    critical_list = critical_df[cols].head(20).to_dict(orient="records")

    return {
        "behaviors": overall_beh,
        "compliance": overall_comp,
        "heatmap": heatmap_data,
        "critical": critical_list,
        "kpis": {
            "total": len(df),
            "critical_flags": int(df.get("has_critical_flag", pd.Series(False,index=df.index)).sum()),
            "escalation_rate": round(float(df.get("behavior_escalation", pd.Series(0,index=df.index)).mean())*100,1),
            "adverse_events": int(df.get("behavior_adverse_event", pd.Series(0,index=df.index)).sum()),
            "auth_rate": round(float(df.get("behavior_authentication", pd.Series(0,index=df.index)).mean())*100,1),
        }
    }


# ═══════════════════════════════════════════════════════════════
# API — NPS JOIN
# ═══════════════════════════════════════════════════════════════

@app.get("/api/nps")
async def api_nps(market: str = "All", channel: str = "All"):
    _ensure_demo()
    df = _store["interactions"].copy()
    kpi = _store["kpi_data"]

    if market != "All": df = df[df["market"]==market]
    if channel != "All": df = df[df["channel"]==channel]

    avg_nps = float(kpi["nps"].mean()) if "nps" in kpi.columns else 30.0

    topic_stats = (
        df.groupby(["topic_label","l1_domain"])
        .agg(volume=("topic_label","count"), avg_sentiment=("sentiment_score","mean"), dsat_rate=("dsat_flag","mean"))
        .reset_index()
    )
    topic_stats["topic_nps"] = (topic_stats["avg_sentiment"] * 50 + avg_nps).round(1)
    topic_stats["avg_sentiment"] = topic_stats["avg_sentiment"].round(3)
    topic_stats["dsat_pct"] = (topic_stats["dsat_rate"]*100).round(1)

    detractors = df[df["sentiment"]=="neg"].groupby("topic_label").size().reset_index(name="count").nlargest(8,"count").to_dict(orient="records")
    promoters = df[df["sentiment"]=="pos"].groupby("topic_label").size().reset_index(name="count").nlargest(8,"count").to_dict(orient="records")

    nps_by_market = kpi.groupby("market")["nps"].mean().round(1).to_dict() if "market" in kpi.columns else {}

    return {
        "topics": topic_stats.to_dict(orient="records"),
        "detractors": detractors,
        "promoters": promoters,
        "nps_by_market": [{"market":k,"nps":v} for k,v in nps_by_market.items()],
        "avg_nps": round(avg_nps,1),
    }


# ═══════════════════════════════════════════════════════════════
# API — DIGITAL / AUTOMATION
# ═══════════════════════════════════════════════════════════════

@app.get("/api/digital")
async def api_digital():
    _ensure_demo()
    df = _store["interactions"]
    auto = _store["automation"]

    channels = df["channel"].value_counts()
    digital = sum(channels.get(c,0) for c in ["chat","review","nps"])
    manned  = sum(channels.get(c,0) for c in ["call","email"])

    digital_df = df[df["l1_domain"]=="Digital Experience"] if "l1_domain" in df.columns else pd.DataFrame()
    digital_topics = (
        digital_df.groupby("topic_label").agg(volume=("topic_label","count"),avg_sentiment=("sentiment_score","mean"))
        .reset_index().sort_values("volume",ascending=False).to_dict(orient="records")
    ) if not digital_df.empty else []

    auto_list = []
    if auto is not None and not auto.empty:
        auto_list = auto.head(15).to_dict(orient="records")

    return {
        "manned": int(manned),
        "digital": int(digital),
        "automation_candidates": auto_list,
        "digital_topics": digital_topics,
    }


# ═══════════════════════════════════════════════════════════════
# API — ALERTS
# ═══════════════════════════════════════════════════════════════

@app.get("/api/alerts")
async def api_alerts():
    _ensure_demo()
    alerts = _store["alerts"]
    if alerts is None or alerts.empty:
        return {"alerts": []}
    return {"alerts": alerts.to_dict(orient="records")}


# ═══════════════════════════════════════════════════════════════
# API — SEARCH
# ═══════════════════════════════════════════════════════════════

@app.get("/api/search")
async def api_search(q: str, top_k: int = 10):
    _ensure_demo()
    df = _store["interactions"]
    if not q:
        return {"results": []}

    # Simple text search fallback (embedding search needs model download)
    mask = df["pii_masked_text"].str.contains(q, case=False, na=False)
    if mask.sum() == 0:
        mask = df["raw_text"].str.contains(q, case=False, na=False)

    cols = [c for c in ["interaction_id","created_ts","channel","lang","market",
                        "pii_masked_text","sentiment","sentiment_score","topic_label"] if c in df.columns]
    results = df[mask][cols].head(top_k)
    return {"results": results.to_dict(orient="records"), "total": int(mask.sum())}


# ═══════════════════════════════════════════════════════════════
# API — TAXONOMY
# ═══════════════════════════════════════════════════════════════

@app.get("/api/taxonomy")
async def api_taxonomy():
    themes = []
    for key, data in TAXONOMY.items():
        themes.append({
            "key": key,
            "label_en": data.get("labels",{}).get("en", key),
            "label_cs": data.get("labels",{}).get("cs", key),
            "level1": data.get("level1",""),
            "level2": data.get("level2",""),
            "level3": data.get("level3",""),
            "seeds_en": data.get("seeds",{}).get("en",[]),
            "seeds_cs": data.get("seeds",{}).get("cs",[]),
            "automation_eligible": data.get("automation_eligible",False),
            "behaviors": data.get("behaviors",[]),
            "compliance": data.get("compliance",[]),
        })
    return {"themes": themes, "version": TAX_DATA.get("taxonomy_version","1.0.0")}


# ═══════════════════════════════════════════════════════════════
# API — UPLOAD + PIPELINE
# ═══════════════════════════════════════════════════════════════

@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...), source_system: str = Form("generic")):
    content = await file.read()
    ingest = pipe("ingest")
    df = ingest.ingest_bytes(content, file.filename, source_system)
    df = _mock_analysis(df)
    _store["interactions"] = df
    return {"rows": len(df), "columns": df.columns.tolist(), "message": f"Loaded {len(df)} interactions"}


@app.post("/api/pipeline/run")
async def api_run_pipeline():
    """Run lightweight pipeline (PII + sentiment + behaviors + compliance + anomalies)."""
    df = _store.get("interactions")
    if df is None:
        raise HTTPException(400, "No data loaded. Upload or use demo data first.")

    steps = []
    try:
        df = pipe("pii").process_dataframe(df)
        steps.append({"name": "PII Redaction", "status": "complete"})
        df = pipe("sentiment").process_dataframe(df)
        steps.append({"name": "Sentiment Analysis", "status": "complete"})
        df = pipe("behaviors").process_dataframe(df)
        steps.append({"name": "Behavior Detection", "status": "complete"})
        df = pipe("compliance").process_dataframe(df)
        steps.append({"name": "Compliance Flags", "status": "complete"})
        alerts_df = pipe("anomalies").detect_anomalies(df)
        steps.append({"name": "Anomaly Detection", "status": "complete"})
        _store["interactions"] = df
        _store["alerts"] = alerts_df if not alerts_df.empty else _store["alerts"]
        _store["automation"] = pipe("impact").get_automation_candidates(df, TAXONOMY)
    except Exception as e:
        steps.append({"name": "Error", "status": "error", "message": str(e)})

    return {"steps": steps, "rows": len(df)}


# ═══════════════════════════════════════════════════════════════
# API — EXPORT
# ═══════════════════════════════════════════════════════════════

@app.get("/api/export/interactions")
async def export_interactions():
    _ensure_demo()
    df = _store["interactions"].drop(columns=["embedding","pii_audit"], errors="ignore")
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    df.to_csv(tmp.name, index=False, encoding="utf-8-sig")
    return FileResponse(tmp.name, filename="cii_interactions.csv", media_type="text/csv")


@app.get("/api/export/capa")
async def export_capa():
    _ensure_demo()
    df = _store["interactions"]
    alerts = _store["alerts"]
    rows = []
    if alerts is not None and not alerts.empty:
        for _, a in alerts.iterrows():
            rows.append({"type":"alert","priority":"high","description":a.get("explanation",""),
                        "recommended_action":a.get("playbook_en",""),"status":"open"})
    if "has_critical_flag" in df.columns:
        for _, r in df[df["has_critical_flag"]].head(20).iterrows():
            rows.append({"type":"compliance","priority":"critical",
                        "description":f"Critical flags: {r.get('critical_flags',[])}",
                        "recommended_action":"Immediate SOP review","status":"open"})
    capa_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    capa_df.to_csv(tmp.name, index=False)
    return FileResponse(tmp.name, filename="cii_capa.csv", media_type="text/csv")


@app.get("/api/export/automation-backlog")
async def export_auto():
    _ensure_demo()
    auto = _store["automation"]
    if auto is None or auto.empty:
        raise HTTPException(404, "No automation data")
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    auto.to_csv(tmp.name, index=False)
    return FileResponse(tmp.name, filename="cii_automation_backlog.csv", media_type="text/csv")


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("webapp:app", host="0.0.0.0", port=port, reload=False)
