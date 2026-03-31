"""
utils/export.py
---------------
Export QC decision log as TSV / CSV.

Decisions are stored in st.session_state.qc_decisions and accumulated
across subjects during a session. This module renders the export UI
and produces downloadable bytes.
"""

from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
import streamlit as st


def _decisions_to_dataframe() -> pd.DataFrame:
    """Convert session QC decisions dict to a tidy DataFrame."""
    decisions: dict = st.session_state.get("qc_decisions", {})
    if not decisions:
        return pd.DataFrame(
            columns=["subject_id", "decision", "notes", "timestamp"]
        )

    rows = []
    for subject_id, meta in decisions.items():
        rows.append(
            {
                "subject_id": subject_id,
                "decision": meta.get("decision", "UNREVIEWED"),
                "notes": meta.get("notes", ""),
                "timestamp": meta.get("timestamp", ""),
            }
        )
    return pd.DataFrame(rows)


def render_export_section(subject_id: str) -> None:
    """Render the QC export tab.

    Parameters
    ----------
    subject_id:
        Currently active subject identifier shown in the sidebar.
    """
    st.markdown("### ↓ Export QC Decisions")
    st.caption(
        "All QC pass/fail decisions made this session are logged below. "
        "Export as TSV for downstream Nipoppy integration."
    )

    st.divider()

    # ── Manual decision entry for current subject ─────────────────────────
    st.markdown(f"#### Decision for `{subject_id}`")

    col_dec, col_notes = st.columns([1, 2])

    with col_dec:
        decision = st.selectbox(
            "QC Decision",
            options=["UNREVIEWED", "PASS", "FAIL", "UNCERTAIN"],
            key="export_decision_select",
        )

    with col_notes:
        notes = st.text_input(
            "Notes (optional)",
            placeholder="e.g. motion artifacts in temporal lobe",
            key="export_notes_input",
        )

    if st.button("💾  Save Decision", key="save_qc_decision"):
        st.session_state.qc_decisions[subject_id] = {
            "decision": decision,
            "notes": notes,
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        st.success(f"Decision saved: **{decision}** for `{subject_id}`")

    st.divider()

    # ── Decision log table ────────────────────────────────────────────────
    st.markdown("#### Session QC Log")
    df = _decisions_to_dataframe()

    if df.empty:
        st.info("No decisions recorded yet. Use the controls above to log QC outcomes.")
        return

    # Colour-code the decision column
    def _style_decision(val: str) -> str:
        colours = {
            "PASS": "color: #3ddc84",
            "FAIL": "color: #ff6b6b",
            "UNCERTAIN": "color: #ffa726",
            "UNREVIEWED": "color: #8fa3bc",
        }
        return colours.get(val, "")

    styled_df = df.style.applymap(_style_decision, subset=["decision"])
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.markdown(f"**{len(df)} subjects** · "
                f"{(df.decision == 'PASS').sum()} PASS · "
                f"{(df.decision == 'FAIL').sum()} FAIL · "
                f"{(df.decision == 'UNCERTAIN').sum()} UNCERTAIN")

    st.divider()

    # ── Download buttons ──────────────────────────────────────────────────
    col_tsv, col_csv = st.columns(2)

    tsv_bytes = df.to_csv(sep="\t", index=False).encode("utf-8")
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    with col_tsv:
        st.download_button(
            label="⬇  Download TSV  (Nipoppy-compatible)",
            data=tsv_bytes,
            file_name=f"qc_decisions_{timestamp_str}.tsv",
            mime="text/tab-separated-values",
            use_container_width=True,
        )

    with col_csv:
        st.download_button(
            label="⬇  Download CSV",
            data=csv_bytes,
            file_name=f"qc_decisions_{timestamp_str}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Clear session log ─────────────────────────────────────────────────
    st.divider()
    if st.button("🗑  Clear Session Log", key="clear_qc_log"):
        st.session_state.qc_decisions = {}
        st.rerun()
