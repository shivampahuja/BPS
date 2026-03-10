"""
Impact analysis pipeline.
Estimates contribution of themes/behaviors to NPS/CSAT/cost metrics.
Uses regularized regression or permutation importance proxy.
"""
import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ImpactPipeline:
    """
    Estimates business impact of CII themes and behaviors on KPIs.
    """

    def __init__(self, config: dict):
        self.config = config
        impact_cfg = config.get("analytics", {}).get("impact", {})
        self.model_type = impact_cfg.get("model", "ridge")
        self.alpha = impact_cfg.get("alpha", 1.0)
        self.test_size = impact_cfg.get("test_size", 0.2)
        self._models: dict = {}

    def compute_topic_impact(
        self,
        interactions: pd.DataFrame,
        kpi_df: Optional[pd.DataFrame] = None,
        target_kpi: str = "nps",
    ) -> pd.DataFrame:
        """
        Compute impact of each topic on a target KPI.
        If kpi_df is provided, join on date/market; otherwise use sentiment as proxy.
        """
        if interactions.empty:
            return pd.DataFrame()

        if kpi_df is not None and not kpi_df.empty and target_kpi in kpi_df.columns:
            return self._compute_joined_impact(interactions, kpi_df, target_kpi)
        else:
            return self._compute_sentiment_proxy_impact(interactions, target_kpi)

    def _compute_joined_impact(
        self,
        interactions: pd.DataFrame,
        kpi_df: pd.DataFrame,
        target_kpi: str,
    ) -> pd.DataFrame:
        """Join interaction topic features with KPI data and fit model."""
        try:
            interactions = interactions.copy()
            interactions["_date"] = pd.to_datetime(
                interactions.get("created_ts", pd.Series(dtype="datetime64[ns]")),
                errors="coerce"
            ).dt.date

            # Aggregate: count of each topic per date/market
            if "topic_label" not in interactions.columns:
                return self._compute_sentiment_proxy_impact(interactions, target_kpi)

            topic_daily = interactions.groupby(["_date", "topic_label"]).size().reset_index(name="count")
            topic_wide = topic_daily.pivot_table(
                index="_date", columns="topic_label", values="count", fill_value=0
            )
            topic_wide.columns = [f"topic_{c}" for c in topic_wide.columns]
            topic_wide = topic_wide.reset_index()

            kpi_df = kpi_df.copy()
            kpi_df["_date"] = pd.to_datetime(kpi_df.get("date", kpi_df.iloc[:, 0]), errors="coerce").dt.date

            merged = topic_wide.merge(kpi_df[["_date", target_kpi]], on="_date", how="inner")

            if len(merged) < 5:
                logger.warning("Insufficient joined data for impact analysis")
                return self._compute_sentiment_proxy_impact(interactions, target_kpi)

            feature_cols = [c for c in merged.columns if c.startswith("topic_")]
            X = merged[feature_cols].values
            y = merged[target_kpi].fillna(0).values

            importances = self._fit_and_get_importance(X, y, feature_cols)
            return importances

        except Exception as e:
            logger.error(f"Joined impact failed: {e}")
            return self._compute_sentiment_proxy_impact(interactions, target_kpi)

    def _compute_sentiment_proxy_impact(
        self,
        interactions: pd.DataFrame,
        target_kpi: str,
    ) -> pd.DataFrame:
        """
        Use sentiment score as proxy KPI target when NPS data not available.
        """
        if "topic_label" not in interactions.columns or "sentiment_score" not in interactions.columns:
            return pd.DataFrame()

        topic_impact = (
            interactions.groupby("topic_label")
            .agg(
                volume=("sentiment_score", "count"),
                avg_sentiment=("sentiment_score", "mean"),
                dsat_rate=("dsat_flag", "mean") if "dsat_flag" in interactions.columns else ("sentiment_score", lambda x: (x < -0.3).mean()),
            )
            .reset_index()
        )

        # Estimate impact score: volume * negative sentiment
        topic_impact["impact_score"] = (
            topic_impact["volume"] * topic_impact["avg_sentiment"].abs()
        )
        topic_impact["direction"] = topic_impact["avg_sentiment"].apply(
            lambda s: "negative" if s < -0.1 else "positive" if s > 0.1 else "neutral"
        )
        topic_impact["kpi"] = f"{target_kpi}_proxy"
        topic_impact["method"] = "sentiment_proxy"

        return topic_impact.sort_values("impact_score", ascending=False)

    def _fit_and_get_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list,
    ) -> pd.DataFrame:
        """Fit regularized model and extract feature importances."""
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split

            if len(X) < 5:
                return pd.DataFrame()

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            if self.model_type == "ridge":
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=self.alpha)
            elif self.model_type == "lasso":
                from sklearn.linear_model import Lasso
                model = Lasso(alpha=self.alpha)
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=50, random_state=42)

            model.fit(X_scaled, y)

            if hasattr(model, "coef_"):
                importances = model.coef_
            else:
                importances = model.feature_importances_

            result = pd.DataFrame({
                "topic_label": [f.replace("topic_", "") for f in feature_names],
                "importance": importances,
                "abs_importance": np.abs(importances),
                "direction": ["positive" if v > 0 else "negative" for v in importances],
                "method": self.model_type,
            })

            return result.sort_values("abs_importance", ascending=False)

        except Exception as e:
            logger.error(f"Model fitting failed: {e}")
            return pd.DataFrame()

    def compute_behavior_impact(
        self,
        interactions: pd.DataFrame,
        target_kpi: str = "sentiment_score",
    ) -> pd.DataFrame:
        """
        Compute impact of behavioral signals on satisfaction proxy.
        """
        if interactions.empty:
            return pd.DataFrame()

        behavior_cols = [c for c in interactions.columns if c.startswith("behavior_")]
        if not behavior_cols or target_kpi not in interactions.columns:
            return pd.DataFrame()

        rows = []
        for b_col in behavior_cols:
            behavior = b_col.replace("behavior_", "")
            has_behavior = interactions[b_col].astype(bool)
            without = interactions.loc[~has_behavior, target_kpi]
            with_behavior = interactions.loc[has_behavior, target_kpi]

            if len(with_behavior) < 2 or len(without) < 2:
                continue

            diff = float(with_behavior.mean() - without.mean())
            rows.append({
                "behavior": behavior,
                "with_behavior_avg": round(float(with_behavior.mean()), 4),
                "without_behavior_avg": round(float(without.mean()), 4),
                "impact": round(diff, 4),
                "count_with": int(len(with_behavior)),
                "direction": "positive" if diff > 0 else "negative",
            })

        return pd.DataFrame(rows).sort_values("impact", ascending=False) if rows else pd.DataFrame()

    def get_automation_candidates(
        self,
        interactions: pd.DataFrame,
        taxonomy: dict,
    ) -> pd.DataFrame:
        """
        Identify topics eligible for automation based on volume + complexity.
        """
        if interactions.empty or "topic_label" not in interactions.columns:
            return pd.DataFrame()

        topic_stats = (
            interactions.groupby("topic_label")
            .agg(
                volume=("topic_label", "count"),
                avg_sentiment=("sentiment_score", "mean") if "sentiment_score" in interactions.columns else ("topic_label", lambda x: 0),
            )
            .reset_index()
        )

        # Check taxonomy for automation_eligible flag
        automation_flags = {}
        for theme_key, theme_data in taxonomy.items():
            label_en = theme_data.get("labels", {}).get("en", "")
            automation_flags[label_en] = theme_data.get("automation_eligible", False)

        topic_stats["automation_eligible"] = topic_stats["topic_label"].map(
            lambda label: automation_flags.get(label, False)
        )

        # Compute automation score: high volume + high automation_eligible
        max_vol = max(topic_stats["volume"].max(), 1)
        topic_stats["automation_score"] = (
            topic_stats["volume"] / max_vol * topic_stats["automation_eligible"].astype(float)
        )

        return topic_stats.sort_values("automation_score", ascending=False)
