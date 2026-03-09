"""
Anomaly detection pipeline.
Uses rolling z-score and EWM on theme volumes, sentiment, behaviors, and KPIs.
Generates alerts with explanations and playbook suggestions.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PLAYBOOKS = {
    "volume_spike": {
        "en": "High volume spike detected. Consider: 1) Check for product/service incident, 2) Increase staffing, 3) Prepare FAQ/bot responses",
        "cs": "Detekován spike objemu. Zvažte: 1) Zkontrolujte produkt/službu, 2) Zvyšte obsazení, 3) Připravte odpovědi FAQ/bota",
    },
    "negative_sentiment_spike": {
        "en": "Negative sentiment spike. Consider: 1) Review recent changes, 2) Trigger CAPA process, 3) Prepare callbacks for top detractors",
        "cs": "Spike negativního sentimentu. Zvažte: 1) Zkontrolujte nedávné změny, 2) Spusťte CAPA, 3) Připravte zpětná volání kritiků",
    },
    "dsat_spike": {
        "en": "DSAT rate spike. Immediate action: 1) Alert CSC leadership, 2) Review top complaint topics, 3) Schedule quality calibration",
        "cs": "Spike míry DSAT. Okamžitá akce: 1) Informujte vedení CSC, 2) Přezkoumejte nejčastější témata stížností, 3) Naplánujte kalibraci kvality",
    },
    "behavior_drop": {
        "en": "Behavior compliance drop. Actions: 1) Targeted coaching, 2) Refresh SOP training, 3) Increase monitoring frequency",
        "cs": "Pokles souladu chování. Akce: 1) Cílený koučink, 2) Obnovit školení SOP, 3) Zvýšit frekvenci monitorování",
    },
    "topic_emergence": {
        "en": "New topic emerging. Actions: 1) Investigate root cause, 2) Prepare agent guidance, 3) Consider digital self-service",
        "cs": "Nové téma se objevuje. Akce: 1) Prošetřete příčinu, 2) Připravte pokyny pro agenty, 3) Zvažte digitální samoobsluhu",
    },
    "kpi_drop": {
        "en": "KPI metric drop. Actions: 1) Root cause analysis, 2) Service quality review, 3) Operational improvements",
        "cs": "Pokles KPI metriky. Akce: 1) Analýza příčin, 2) Přezkoumání kvality, 3) Operativní zlepšení",
    },
}


class AnomalyPipeline:
    """
    Anomaly and trend detection on aggregated CII metrics.
    Generates timestamped alerts with explanations.
    """

    def __init__(self, config: dict):
        self.config = config
        anomaly_cfg = config.get("analytics", {}).get("anomaly", {})
        self.window_days = anomaly_cfg.get("window_days", 7)
        self.z_threshold = anomaly_cfg.get("z_score_threshold", 2.5)
        self.ewm_span = anomaly_cfg.get("ewm_span", 7)
        self.min_volume = anomaly_cfg.get("min_volume", 10)

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        date_col: str = "created_ts",
        group_cols: Optional[list] = None,
    ) -> pd.DataFrame:
        """
        Detect anomalies in interaction data.
        Returns DataFrame of alerts.
        """
        if df.empty or date_col not in df.columns:
            return pd.DataFrame()

        alerts = []
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df["_date"] = df[date_col].dt.date

        # Overall volume anomalies
        daily_counts = df.groupby("_date").size().reset_index(name="volume")
        daily_counts["_date"] = pd.to_datetime(daily_counts["_date"])
        alerts += self._detect_metric_anomalies(
            daily_counts, "_date", "volume", "volume_spike"
        )

        # Sentiment anomalies
        if "sentiment_score" in df.columns:
            daily_sentiment = df.groupby("_date")["sentiment_score"].mean().reset_index()
            daily_sentiment.columns = ["_date", "avg_sentiment"]
            daily_sentiment["_date"] = pd.to_datetime(daily_sentiment["_date"])
            alerts += self._detect_metric_anomalies(
                daily_sentiment, "_date", "avg_sentiment",
                "negative_sentiment_spike", direction="down"
            )

        # DSAT anomalies
        if "dsat_flag" in df.columns:
            daily_dsat = df.groupby("_date")["dsat_flag"].mean().reset_index()
            daily_dsat.columns = ["_date", "dsat_rate"]
            daily_dsat["_date"] = pd.to_datetime(daily_dsat["_date"])
            alerts += self._detect_metric_anomalies(
                daily_dsat, "_date", "dsat_rate", "dsat_spike"
            )

        # Topic volume anomalies
        if "topic_label" in df.columns:
            for topic in df["topic_label"].dropna().unique():
                topic_df = df[df["topic_label"] == topic]
                topic_daily = topic_df.groupby("_date").size().reset_index(name="volume")
                topic_daily["_date"] = pd.to_datetime(topic_daily["_date"])

                if len(topic_daily) < 3:
                    continue

                topic_alerts = self._detect_metric_anomalies(
                    topic_daily, "_date", "volume", "topic_emergence"
                )
                for alert in topic_alerts:
                    alert["dimension_json"] = {"topic": topic}
                alerts += topic_alerts

        # Behavior drop anomalies
        for behavior in ["behavior_empathy", "behavior_authentication", "behavior_resolution_confirmation"]:
            if behavior in df.columns:
                daily_beh = df.groupby("_date")[behavior].mean().reset_index()
                daily_beh.columns = ["_date", "rate"]
                daily_beh["_date"] = pd.to_datetime(daily_beh["_date"])
                alerts += self._detect_metric_anomalies(
                    daily_beh, "_date", "rate", "behavior_drop", direction="down"
                )

        if not alerts:
            return pd.DataFrame()

        alerts_df = pd.DataFrame(alerts)
        alerts_df["alert_id"] = [
            f"ALT-{i+1:04d}" for i in range(len(alerts_df))
        ]
        alerts_df["status"] = "open"
        return alerts_df.sort_values("date", ascending=False)

    def _detect_metric_anomalies(
        self,
        daily: pd.DataFrame,
        date_col: str,
        metric_col: str,
        alert_type: str,
        direction: str = "up",
    ) -> list:
        """Detect anomalies using rolling z-score."""
        if len(daily) < max(3, self.window_days):
            return []

        daily = daily.sort_values(date_col).copy()
        series = daily[metric_col].fillna(0)

        # Rolling statistics
        rolling_mean = series.rolling(window=self.window_days, min_periods=3).mean()
        rolling_std = series.rolling(window=self.window_days, min_periods=3).std()

        # Z-scores
        z_scores = (series - rolling_mean) / (rolling_std + 1e-8)

        # EWM for trend
        ewm = series.ewm(span=self.ewm_span).mean()
        ewm_diff = ewm.diff()

        alerts = []
        for i, (z, val, date) in enumerate(
            zip(z_scores, series, daily[date_col])
        ):
            if pd.isna(z) or pd.isna(val):
                continue

            triggered = False
            if direction == "up" and z > self.z_threshold:
                triggered = True
            elif direction == "down" and z < -self.z_threshold:
                triggered = True

            if triggered:
                magnitude = float(abs(z))
                alerts.append({
                    "date": date,
                    "metric": metric_col,
                    "alert_type": alert_type,
                    "direction": direction,
                    "value": round(float(val), 4),
                    "z_score": round(magnitude, 3),
                    "magnitude": round(magnitude, 3),
                    "dimension_json": {},
                    "explanation": self._generate_explanation(
                        alert_type, metric_col, direction, float(val),
                        float(rolling_mean.iloc[i]) if not pd.isna(rolling_mean.iloc[i]) else 0,
                        magnitude
                    ),
                    "playbook_en": PLAYBOOKS.get(alert_type, {}).get("en", ""),
                    "playbook_cs": PLAYBOOKS.get(alert_type, {}).get("cs", ""),
                    "notes": "",
                })

        return alerts

    def _generate_explanation(
        self,
        alert_type: str,
        metric: str,
        direction: str,
        value: float,
        baseline: float,
        z_score: float,
    ) -> str:
        direction_str = "spike" if direction == "up" else "drop"
        pct_change = abs((value - baseline) / (baseline + 1e-8)) * 100
        return (
            f"{metric} {direction_str}: current={value:.3f}, "
            f"baseline={baseline:.3f}, "
            f"change={pct_change:.1f}%, z-score={z_score:.2f}"
        )

    def detect_from_kpi_data(self, kpi_df: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies from KPI daily data."""
        if kpi_df.empty:
            return pd.DataFrame()

        alerts = []
        kpi_metrics = ["nps", "csat", "ces", "aht_sec", "repeat_rate", "replacements"]

        for metric in kpi_metrics:
            if metric not in kpi_df.columns:
                continue

            daily = kpi_df[["date", metric]].dropna().copy()
            daily.columns = ["_date", "value"]
            daily["_date"] = pd.to_datetime(daily["_date"])

            direction = "down" if metric in ["nps", "csat", "ces"] else "up"
            metric_alerts = self._detect_metric_anomalies(
                daily, "_date", "value", "kpi_drop", direction=direction
            )
            for alert in metric_alerts:
                alert["metric"] = metric
                alert["dimension_json"] = {"kpi": metric}
            alerts += metric_alerts

        if not alerts:
            return pd.DataFrame()

        alerts_df = pd.DataFrame(alerts)
        alerts_df["alert_id"] = [f"KPI-{i+1:04d}" for i in range(len(alerts_df))]
        alerts_df["status"] = "open"
        return alerts_df

    def backtest(
        self,
        df: pd.DataFrame,
        seeded_spikes: list[dict],
        date_col: str = "created_ts",
    ) -> dict:
        """
        Backtest anomaly detection against known seeded spikes.
        Returns: {recall, false_positive_rate, detected, missed, false_positives}
        """
        detected_alerts = self.detect_anomalies(df, date_col)

        if detected_alerts.empty or not seeded_spikes:
            return {"recall": 0.0, "fpr": 0.0, "detected": 0, "missed": len(seeded_spikes), "fp": 0}

        detected_dates = set(pd.to_datetime(detected_alerts["date"]).dt.date.tolist())
        seeded_dates = set()

        for spike in seeded_spikes:
            spike_date = pd.to_datetime(spike.get("date")).date()
            window = spike.get("window_days", 1)
            for d in range(-window, window + 1):
                seeded_dates.add(spike_date + timedelta(days=d))

        true_positives = len(detected_dates & seeded_dates)
        false_positives = len(detected_dates - seeded_dates)
        false_negatives = len(seeded_dates - detected_dates)

        recall = true_positives / max(1, true_positives + false_negatives)
        fpr = false_positives / max(1, len(detected_dates))

        return {
            "recall": round(recall, 4),
            "false_positive_rate": round(fpr, 4),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        }
