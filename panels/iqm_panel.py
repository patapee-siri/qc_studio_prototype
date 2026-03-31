"""
panels/iqm_panel.py
-------------------
Interactive IQM dashboard panel.

Renders Plotly-based charts for MRIQC Image Quality Metrics:
- Per-metric gauge / bar charts with reference-range shading
- Group-level scatter matrix for multi-subject TSVs
- Automatic pass/warn/fail badges using heuristic thresholds
- Per-row QC flag column added to the dataframe view
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from loaders.iqm_loader import (
    IQM_REFERENCE_RANGES,
    load_iqm,
    qc_flag_single_row,
)


# ── Plotly dark theme base ────────────────────────────────────────────────────
_PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0a0e1a",
    plot_bgcolor="#0d1221",
    font=dict(family="JetBrains Mono, monospace", color="#8fa3bc", size=11),
    margin=dict(l=40, r=20, t=40, b=30),
    colorway=["#4d9de0", "#3ddc84", "#ffa726", "#ff6b6b", "#c77dff"],
)

_STATUS_HTML = {
    "PASS": '<span class="badge-pass">PASS</span>',
    "FAIL": '<span class="badge-fail">FAIL</span>',
    "WARN": '<span class="badge-warn">WARN</span>',
    "UNREVIEWED": '<span style="color:#5a7a99;font-family:monospace;font-size:0.72rem;">—</span>',
}


def _metric_bar_chart(df: pd.DataFrame, col: str) -> go.Figure:
    """Create a bar chart for a single IQM column across subjects/rows."""
    ref = IQM_REFERENCE_RANGES.get(col, {})
    label = ref.get("label", col.upper())
    unit = ref.get("unit", "")
    direction = ref.get("direction", "higher_better")

    values = df[col].dropna()
    if values.empty:
        fig = go.Figure()
        fig.update_layout(**_PLOTLY_LAYOUT, title=f"{label} — no data")
        return fig

    # ── Bar colours based on direction + good threshold ───────────────────
    good_min = ref.get("good_min", None)
    good_max = ref.get("good_max", None)

    bar_colors = []
    for v in df[col]:
        if pd.isna(v):
            bar_colors.append("#2a4a72")
            continue
        if direction == "higher_better" and good_min is not None:
            bar_colors.append("#3ddc84" if v >= good_min else
                              "#ffa726" if v >= good_min * 0.7 else "#ff6b6b")
        elif direction == "lower_better" and good_max is not None:
            bar_colors.append("#3ddc84" if v <= good_max else
                              "#ffa726" if v <= good_max * 1.4 else "#ff6b6b")
        else:
            bar_colors.append("#4d9de0")

    # X axis: subject index or column name
    x_labels = (
        df.get("subject_id", df.get("bids_name", pd.RangeIndex(len(df))))
        .astype(str).tolist()
        if "subject_id" in df.columns or "bids_name" in df.columns
        else [str(i) for i in range(len(df))]
    )

    fig = go.Figure(
        go.Bar(
            x=x_labels,
            y=df[col].tolist(),
            marker_color=bar_colors,
            text=[f"{v:.3f}" if not pd.isna(v) else "—" for v in df[col]],
            textposition="outside",
            textfont=dict(size=9),
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.4f}} {unit}<extra></extra>",
        )
    )

    # ── Reference-range shading ───────────────────────────────────────────
    y_range_min = float(ref.get("min", values.min() * 0.8))
    y_range_max = float(ref.get("max", values.max() * 1.2))

    if good_min is not None:
        fig.add_hrect(
            y0=good_min, y1=y_range_max,
            fillcolor="rgba(61,220,132,0.04)",
            line_width=0, annotation_text="good",
            annotation_font=dict(size=9, color="#3ddc84"),
            annotation_position="top right",
        )
    if good_max is not None:
        fig.add_hrect(
            y0=y_range_min, y1=good_max,
            fillcolor="rgba(61,220,132,0.04)",
            line_width=0, annotation_text="good",
            annotation_font=dict(size=9, color="#3ddc84"),
            annotation_position="top right",
        )

    tick_suffix = f" {unit}" if unit else ""
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text=label, font=dict(size=13, color="#e0ecff")),
        yaxis=dict(
            ticksuffix=tick_suffix,
            gridcolor="#1e2d45",
            zeroline=False,
        ),
        xaxis=dict(gridcolor="#1e2d45"),
        showlegend=False,
        height=300,
    )
    return fig


def _scatter_matrix(df: pd.DataFrame, cols: list[str]) -> go.Figure:
    """Build a Plotly scatter matrix for up to 6 IQM columns."""
    n = min(len(cols), 6)
    selected = cols[:n]

    dimensions = []
    for c in selected:
        ref = IQM_REFERENCE_RANGES.get(c, {})
        dimensions.append(
            dict(label=ref.get("label", c.upper()), values=df[c])
        )

    fig = go.Figure(
        go.Splom(
            dimensions=dimensions,
            marker=dict(
                color="#4d9de0",
                size=5,
                opacity=0.7,
                line=dict(width=0),
            ),
            showupperhalf=False,
            diagonal_visible=True,
            text=(
                df["subject_id"].astype(str)
                if "subject_id" in df.columns
                else [str(i) for i in range(len(df))]
            ),
            hovertemplate="<b>%{text}</b><br>%{xaxis.title.text}: %{x:.3f}<br>%{yaxis.title.text}: %{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_PLOTLY_LAYOUT,
        title=dict(text="IQM Scatter Matrix", font=dict(size=13, color="#e0ecff")),
        height=550,
    )
    return fig


def _summary_metrics_row(df: pd.DataFrame, iqm_cols: list[str]) -> None:
    """Render a row of summary metric cards."""
    key_metrics = [c for c in ["snr", "cnr", "efc", "tsnr", "fd_mean"] if c in iqm_cols]
    if not key_metrics:
        key_metrics = iqm_cols[:5]

    cols = st.columns(len(key_metrics))
    for col, metric in zip(cols, key_metrics):
        ref = IQM_REFERENCE_RANGES.get(metric, {})
        label = ref.get("label", metric.upper())
        unit = ref.get("unit", "")
        vals = df[metric].dropna()
        if vals.empty:
            continue
        mean_val = vals.mean()
        direction = ref.get("direction", "higher_better")
        good_min = ref.get("good_min")
        good_max = ref.get("good_max")

        if good_min is not None:
            status = "PASS" if mean_val >= good_min else "FAIL"
        elif good_max is not None:
            status = "PASS" if mean_val <= good_max else "FAIL"
        else:
            status = "—"

        badge = _STATUS_HTML.get(status, "")

        col.markdown(
            f"""
            <div class="qcs-card">
                <div class="qcs-label">{label}</div>
                <div class="qcs-metric">{mean_val:.3f}<span
                  style="font-size:0.75rem;color:#5a7a99;"> {unit}</span></div>
                <div style="margin-top:4px;">{badge}</div>
                <div style="font-size:0.68rem;color:#3a5a7a;margin-top:2px;">
                    n={len(vals)} · σ={vals.std():.3f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_iqm_panel(iqm_file: Optional[object], subject_id: str) -> None:
    """Render the full IQM dashboard panel.

    Parameters
    ----------
    iqm_file:
        Streamlit ``UploadedFile`` or ``None``.
    subject_id:
        Active subject ID for flagging.
    """
    st.markdown("### ◎ IQM Dashboard")

    if iqm_file is None:
        st.info(
            "Upload an MRIQC group TSV (e.g. `group_T1w.tsv` or `group_bold.tsv`) "
            "via the sidebar to explore quality metrics."
        )
        return

    # ── Load dataset ──────────────────────────────────────────────────────
    with st.spinner("Parsing IQM file…"):
        try:
            dataset = load_iqm(iqm_file.getvalue(), iqm_file.name)
        except ValueError as exc:
            st.error(f"Cannot load IQM file: {exc}")
            return

    df = dataset.df
    iqm_cols = dataset.iqm_columns

    # Show any parse warnings
    for w in dataset.warnings:
        st.warning(w)

    # ── Header info ───────────────────────────────────────────────────────
    st.markdown(
        f"**`{iqm_file.name}`** · {len(df)} subjects · "
        f"modality: `{dataset.modality}` · "
        f"{len(iqm_cols)} IQM columns recognised"
    )
    st.caption(
        "Reference ranges based on Esteban et al. (2017) MRIQC + community norms."
    )

    st.divider()

    # ── Summary metric cards ──────────────────────────────────────────────
    _summary_metrics_row(df, iqm_cols)

    st.divider()

    # ── QC flag column ────────────────────────────────────────────────────
    st.markdown("#### Per-Subject QC Flags")
    df_flagged = df.copy()
    df_flagged["qc_flag"] = df_flagged.apply(qc_flag_single_row, axis=1)

    # Colour style function
    def _style_flag(val: str) -> str:
        return {
            "PASS": "color: #3ddc84; font-weight: 600",
            "FAIL": "color: #ff6b6b; font-weight: 600",
            "WARN": "color: #ffa726; font-weight: 600",
        }.get(val, "")

    display_cols = (
        ["subject_id"] if "subject_id" in df_flagged.columns else []
    ) + iqm_cols[:8] + ["qc_flag"]

    styled = df_flagged[display_cols].style.applymap(
        _style_flag, subset=["qc_flag"]
    ).format(
        {c: "{:.4f}" for c in iqm_cols[:8] if c in df_flagged.columns},
        na_rep="—"
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    n_pass = (df_flagged["qc_flag"] == "PASS").sum()
    n_fail = (df_flagged["qc_flag"] == "FAIL").sum()
    n_warn = (df_flagged["qc_flag"] == "WARN").sum()
    st.markdown(
        f"Auto-flagged: &nbsp;"
        f'<span class="badge-pass">PASS {n_pass}</span> &nbsp;'
        f'<span class="badge-warn">WARN {n_warn}</span> &nbsp;'
        f'<span class="badge-fail">FAIL {n_fail}</span>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Per-metric charts ─────────────────────────────────────────────────
    st.markdown("#### Per-Metric Charts")
    chart_col_options = iqm_cols
    selected_metrics = st.multiselect(
        "Select metrics to plot",
        options=chart_col_options,
        default=chart_col_options[:4],
        key="iqm_metric_select",
    )

    if selected_metrics:
        chart_cols = st.columns(2)
        for i, metric in enumerate(selected_metrics):
            with chart_cols[i % 2]:
                fig = _metric_bar_chart(df, metric)
                st.plotly_chart(fig, use_container_width=True, key=f"iqm_chart_{metric}")

    st.divider()

    # ── Scatter matrix (multi-subject only) ───────────────────────────────
    if len(df) > 1 and len(iqm_cols) >= 2:
        st.markdown("#### IQM Scatter Matrix")
        st.caption("Explore correlations between IQMs across subjects.")
        matrix_cols = st.multiselect(
            "Select IQMs for scatter matrix (2–6)",
            options=iqm_cols,
            default=iqm_cols[:min(4, len(iqm_cols))],
            key="iqm_matrix_select",
        )
        if 2 <= len(matrix_cols) <= 6:
            fig_mat = _scatter_matrix(df, matrix_cols)
            st.plotly_chart(fig_mat, use_container_width=True, key="iqm_scatter_matrix")
        elif len(matrix_cols) > 6:
            st.warning("Select at most 6 IQMs for the scatter matrix.")

    # Cache dataframe for MedGemma panel
    st.session_state.iqm_df = df
