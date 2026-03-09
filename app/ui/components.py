"""
Reusable Gradio UI components for the CII platform.
"""
import json
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# COLOR PALETTE
# ============================================================

COLORS = {
    "positive": "#10b981",
    "neutral": "#6b7280",
    "negative": "#ef4444",
    "primary": "#3b82f6",
    "secondary": "#8b5cf6",
    "warning": "#f59e0b",
    "critical": "#dc2626",
    "background": "#f8fafc",
    "card": "#ffffff",
    "border": "#e2e8f0",
}

SENTIMENT_COLORS = {
    "pos": COLORS["positive"],
    "neu": COLORS["neutral"],
    "neg": COLORS["negative"],
}

SEVERITY_COLORS = {
    "critical": COLORS["critical"],
    "high": COLORS["warning"],
    "medium": COLORS["primary"],
    "low": COLORS["neutral"],
}

DOMAIN_COLORS = {
    "Product Experience": "#3b82f6",
    "Service & Support (CSC)": "#8b5cf6",
    "Digital Experience": "#06b6d4",
    "Commerce & Fulfillment": "#f59e0b",
    "Program & Loyalty": "#10b981",
    "Channel Experience": "#f97316",
    "Policy & Compliance": "#ef4444",
}


# ============================================================
# KPI TILE HTML
# ============================================================

def kpi_tile_html(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_direction: str = "up",
    color: str = COLORS["primary"],
    icon: str = "📊",
) -> str:
    delta_color = COLORS["positive"] if delta_direction == "up" else COLORS["negative"]
    delta_html = f'<span style="color:{delta_color};font-size:0.8rem">{delta}</span>' if delta else ""

    return f"""
    <div style="
        background:{COLORS['card']};
        border:1px solid {COLORS['border']};
        border-radius:12px;
        padding:20px;
        text-align:center;
        box-shadow:0 1px 3px rgba(0,0,0,0.1);
        min-width:140px;
    ">
        <div style="font-size:1.8rem">{icon}</div>
        <div style="font-size:0.85rem;color:#6b7280;margin:4px 0">{label}</div>
        <div style="font-size:1.8rem;font-weight:700;color:{color}">{value}</div>
        {delta_html}
    </div>
    """


def kpi_row_html(kpis: list[dict]) -> str:
    tiles = "".join(kpi_tile_html(**kpi) for kpi in kpis)
    return f'<div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:flex-start">{tiles}</div>'


# ============================================================
# ALERT CARD HTML
# ============================================================

def alert_card_html(alert: dict, lang: str = "en") -> str:
    severity = alert.get("z_score", 0)
    alert_type = alert.get("alert_type", "unknown")
    date = str(alert.get("date", ""))[:10]
    explanation = alert.get("explanation", "")
    playbook_key = f"playbook_{lang}"
    playbook = alert.get(playbook_key, alert.get("playbook_en", ""))
    metric = alert.get("metric", "")

    severity_color = COLORS["critical"] if severity > 3 else COLORS["warning"] if severity > 2 else COLORS["primary"]
    icon = "🚨" if severity > 3 else "⚠️" if severity > 2 else "ℹ️"

    return f"""
    <div style="
        background:{COLORS['card']};
        border-left:4px solid {severity_color};
        border-radius:8px;
        padding:16px;
        margin:8px 0;
        box-shadow:0 1px 3px rgba(0,0,0,0.1);
    ">
        <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="font-weight:600;color:{severity_color}">{icon} {alert_type.replace('_',' ').title()}</span>
            <span style="color:#6b7280;font-size:0.85rem">{date}</span>
        </div>
        <div style="margin:8px 0;color:#374151">{explanation}</div>
        <div style="color:#6b7280;font-size:0.85rem;font-style:italic">{playbook}</div>
    </div>
    """


def alerts_panel_html(alerts_df: pd.DataFrame, lang: str = "en", max_alerts: int = 10) -> str:
    if alerts_df is None or alerts_df.empty:
        return '<div style="color:#6b7280;padding:20px">No active alerts.</div>'

    cards = "".join(
        alert_card_html(row.to_dict(), lang)
        for _, row in alerts_df.head(max_alerts).iterrows()
    )
    return f'<div style="max-height:400px;overflow-y:auto">{cards}</div>'


# ============================================================
# PLOTLY CHARTS
# ============================================================

def sentiment_donut(sentiment_data: dict, lang: str = "en") -> go.Figure:
    """Sentiment distribution donut chart."""
    labels = ["Positive", "Neutral", "Negative"]
    values = [
        sentiment_data.get("positive", 0),
        sentiment_data.get("neutral", 0),
        sentiment_data.get("negative", 0),
    ]
    colors = [COLORS["positive"], COLORS["neutral"], COLORS["negative"]]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.5,
        marker_colors=colors,
        textinfo="percent+label",
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    )])

    fig.update_layout(
        showlegend=True,
        margin=dict(t=40, b=20, l=20, r=20),
        height=300,
        title=dict(text="Sentiment Distribution", font=dict(size=14)),
        paper_bgcolor=COLORS["background"],
    )
    return fig


def topic_bar_chart(
    topic_df: pd.DataFrame,
    x_col: str = "label_en",
    y_col: str = "count",
    color_col: Optional[str] = None,
    title: str = "Topics by Volume",
    orientation: str = "h",
    lang: str = "en",
) -> go.Figure:
    """Horizontal bar chart of topics."""
    if topic_df.empty:
        return empty_chart(title)

    label_col = f"label_{lang}" if f"label_{lang}" in topic_df.columns else x_col

    df_sorted = topic_df.sort_values(y_col, ascending=True).tail(20)

    color_vals = None
    colorscale = None
    if color_col and color_col in df_sorted.columns:
        color_vals = df_sorted[color_col]
        colorscale = [[0, COLORS["negative"]], [0.5, COLORS["neutral"]], [1, COLORS["positive"]]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_sorted[y_col] if orientation == "h" else df_sorted[label_col],
        y=df_sorted[label_col] if orientation == "h" else df_sorted[y_col],
        orientation=orientation,
        marker=dict(
            color=color_vals if color_vals is not None else COLORS["primary"],
            colorscale=colorscale,
            colorbar=dict(title="Sentiment") if color_vals is not None else None,
        ),
        hovertemplate=f"<b>%{{y}}</b><br>{y_col}: %{{x}}<extra></extra>" if orientation == "h" else None,
    ))

    fig.update_layout(
        title=title,
        xaxis_title=y_col.replace("_", " ").title() if orientation == "h" else "",
        yaxis_title="" if orientation == "h" else y_col.replace("_", " ").title(),
        margin=dict(t=50, b=50, l=200, r=20),
        height=max(300, len(df_sorted) * 30 + 100),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
    )
    return fig


def sentiment_trend_chart(
    df: pd.DataFrame,
    date_col: str = "created_ts",
    sentiment_col: str = "sentiment_score",
    title: str = "Sentiment Trend",
) -> go.Figure:
    """Line chart of sentiment over time."""
    if df.empty or date_col not in df.columns:
        return empty_chart(title)

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    daily = df.groupby(df[date_col].dt.date)[sentiment_col].mean().reset_index()
    daily.columns = ["date", "avg_sentiment"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"],
        y=daily["avg_sentiment"],
        mode="lines+markers",
        name="Avg Sentiment",
        line=dict(color=COLORS["primary"], width=2),
        fill="tozeroy",
        fillcolor=f"rgba(59,130,246,0.1)",
    ))

    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["neutral"])
    fig.add_hline(y=-0.3, line_dash="dot", line_color=COLORS["negative"], annotation_text="DSAT threshold")

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Avg Sentiment Score",
        yaxis=dict(range=[-1, 1]),
        height=300,
        margin=dict(t=50, b=50, l=60, r=20),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
    )
    return fig


def volume_trend_chart(
    df: pd.DataFrame,
    date_col: str = "created_ts",
    group_col: Optional[str] = None,
    title: str = "Interaction Volume",
) -> go.Figure:
    """Stacked area or line chart of volume over time."""
    if df.empty or date_col not in df.columns:
        return empty_chart(title)

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["_date"] = df[date_col].dt.date

    fig = go.Figure()

    if group_col and group_col in df.columns:
        for group in df[group_col].dropna().unique():
            group_df = df[df[group_col] == group]
            daily = group_df.groupby("_date").size().reset_index(name="count")
            fig.add_trace(go.Scatter(
                x=daily["_date"],
                y=daily["count"],
                name=str(group),
                mode="lines",
                stackgroup="one",
            ))
    else:
        daily = df.groupby("_date").size().reset_index(name="count")
        fig.add_trace(go.Scatter(
            x=daily["_date"],
            y=daily["count"],
            name="Volume",
            mode="lines+markers",
            line=dict(color=COLORS["primary"], width=2),
            fill="tozeroy",
            fillcolor="rgba(59,130,246,0.1)",
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Count",
        height=300,
        margin=dict(t=50, b=50, l=60, r=20),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
    )
    return fig


def behavior_heatmap(
    df: pd.DataFrame,
    behavior_cols: Optional[list] = None,
    group_col: str = "channel",
    title: str = "Behavior Frequencies by Channel",
) -> go.Figure:
    """Heatmap of behavior frequencies."""
    if df.empty:
        return empty_chart(title)

    if behavior_cols is None:
        behavior_cols = [c for c in df.columns if c.startswith("behavior_")]

    if not behavior_cols or group_col not in df.columns:
        return empty_chart(title)

    pivot = df.groupby(group_col)[behavior_cols].mean() * 100
    pivot.columns = [c.replace("behavior_", "").replace("_", " ").title() for c in pivot.columns]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        text=pivot.values.round(1),
        texttemplate="%{text}%",
        hovertemplate="%{y} — %{x}: %{z:.1f}%<extra></extra>",
        zmin=0,
        zmax=100,
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Behavior",
        yaxis_title=group_col.title(),
        height=max(300, len(pivot) * 50 + 100),
        margin=dict(t=50, b=80, l=120, r=20),
        paper_bgcolor=COLORS["background"],
    )
    return fig


def nps_topic_scatter(
    df: pd.DataFrame,
    x_col: str = "avg_sentiment",
    y_col: str = "volume",
    label_col: str = "topic_label",
    color_col: Optional[str] = None,
    title: str = "Topic NPS vs Volume",
) -> go.Figure:
    """Bubble/scatter chart of topics by sentiment and volume."""
    if df.empty:
        return empty_chart(title)

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        text=label_col if label_col in df.columns else None,
        color=color_col if color_col and color_col in df.columns else None,
        size=y_col,
        size_max=50,
        title=title,
        color_continuous_scale="RdYlGn",
        labels={x_col: "Avg Sentiment", y_col: "Volume"},
    )

    fig.add_vline(x=0, line_dash="dash", line_color=COLORS["neutral"])
    fig.update_traces(textposition="top center", textfont_size=10)

    fig.update_layout(
        height=400,
        margin=dict(t=50, b=50, l=60, r=20),
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
    )
    return fig


def compliance_gauge(
    rate: float,
    title: str = "Compliance Rate",
) -> go.Figure:
    """Gauge chart for compliance rate."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=rate * 100,
        title={"text": title, "font": {"size": 14}},
        number={"suffix": "%", "font": {"size": 24}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": COLORS["primary"]},
            "steps": [
                {"range": [0, 50], "color": COLORS["negative"]},
                {"range": [50, 80], "color": COLORS["warning"]},
                {"range": [80, 100], "color": COLORS["positive"]},
            ],
            "threshold": {
                "line": {"color": COLORS["critical"], "width": 4},
                "thickness": 0.75,
                "value": 90,
            },
        },
    ))

    fig.update_layout(
        height=200,
        margin=dict(t=50, b=20, l=30, r=30),
        paper_bgcolor=COLORS["background"],
    )
    return fig


def impact_waterfall(
    df: pd.DataFrame,
    label_col: str = "topic_label",
    value_col: str = "impact",
    title: str = "Topic NPS Impact",
) -> go.Figure:
    """Waterfall chart for impact analysis."""
    if df.empty:
        return empty_chart(title)

    df_sorted = df.sort_values(value_col, ascending=True).tail(15)

    colors = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in df_sorted[value_col]]

    fig = go.Figure(go.Bar(
        x=df_sorted[label_col],
        y=df_sorted[value_col],
        marker_color=colors,
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    ))

    fig.add_hline(y=0, line_color=COLORS["neutral"])

    fig.update_layout(
        title=title,
        xaxis_title="Topic",
        yaxis_title="Impact",
        height=350,
        margin=dict(t=50, b=120, l=60, r=20),
        xaxis_tickangle=-45,
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
    )
    return fig


def empty_chart(title: str = "", message: str = "No data available") -> go.Figure:
    """Return an empty placeholder chart."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        height=300,
        annotations=[
            dict(
                text=message,
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=14, color=COLORS["neutral"]),
            )
        ],
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["background"],
    )
    return fig


# ============================================================
# PIPELINE STATUS HTML
# ============================================================

def pipeline_status_html(steps: list[dict]) -> str:
    """HTML display for pipeline step statuses."""
    icons = {"pending": "⏳", "running": "🔄", "complete": "✅", "error": "❌"}
    colors = {
        "pending": COLORS["neutral"],
        "running": COLORS["warning"],
        "complete": COLORS["positive"],
        "error": COLORS["negative"],
    }

    rows = ""
    for step in steps:
        name = step.get("name", "")
        status = step.get("status", "pending")
        message = step.get("message", "")
        icon = icons.get(status, "⏳")
        color = colors.get(status, COLORS["neutral"])

        rows += f"""
        <div style="
            display:flex;align-items:center;gap:12px;
            padding:10px;border-radius:8px;
            background:{COLORS['card']};margin:4px 0;
            border:1px solid {COLORS['border']};
        ">
            <span style="font-size:1.2rem">{icon}</span>
            <span style="font-weight:600;color:{color};min-width:180px">{name}</span>
            <span style="color:#6b7280;font-size:0.85rem">{message}</span>
        </div>
        """

    return f'<div style="padding:8px">{rows}</div>'


# ============================================================
# TAXONOMY EDITOR HTML
# ============================================================

def taxonomy_display_html(taxonomy: dict, lang: str = "en") -> str:
    """Display taxonomy as collapsible HTML structure."""
    if not taxonomy:
        return "<div>No taxonomy loaded</div>"

    domains: dict[str, list] = {}
    for theme_key, theme_data in taxonomy.items():
        l1 = theme_data.get("level1", "Unknown")
        if l1 not in domains:
            domains[l1] = []
        domains[l1].append((theme_key, theme_data))

    html = ""
    for domain, themes in sorted(domains.items()):
        color = DOMAIN_COLORS.get(domain, COLORS["primary"])
        html += f"""
        <details style="margin:8px 0">
            <summary style="
                cursor:pointer;padding:10px;
                background:{color}20;border-radius:8px;
                font-weight:600;color:{color};
                border-left:4px solid {color};
            ">{domain} ({len(themes)} themes)</summary>
            <div style="padding:8px 16px">
        """
        for theme_key, theme_data in themes:
            label = theme_data.get("labels", {}).get(lang, theme_key)
            l2 = theme_data.get("level2", "")
            seeds = theme_data.get("seeds", {}).get(lang, [])
            auto = "✅" if theme_data.get("automation_eligible", False) else ""
            html += f"""
            <div style="
                padding:8px;margin:4px 0;
                background:{COLORS['card']};border-radius:6px;
                border:1px solid {COLORS['border']};
            ">
                <span style="font-weight:600">{label}</span>
                <span style="color:#6b7280;font-size:0.8rem;margin-left:8px">{l2}</span>
                {auto}
                <div style="color:#6b7280;font-size:0.8rem;margin-top:4px">
                    Seeds: {', '.join(seeds[:5])}{'...' if len(seeds) > 5 else ''}
                </div>
            </div>
            """
        html += "</div></details>"

    return html
