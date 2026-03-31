"""
utils/session.py
----------------
Centralized Streamlit session-state initialization.

All mutable state used across tabs is declared here so that any module
can safely read / write `st.session_state.<key>` without KeyError.
"""

from __future__ import annotations

import streamlit as st


def init_session_state() -> None:
    """Initialize all session-state keys with safe defaults.

    Called once at app startup (idempotent — existing keys are not reset).
    """
    defaults: dict = {
        # QC decisions: {subject_id: {"decision": str, "notes": str, "timestamp": str}}
        "qc_decisions": {},
        # MedGemma chat history for IQM-explanation mode
        "medgemma_iqm_history": [],
        # MedGemma chat history for image-analysis mode
        "medgemma_img_history": [],
        # Currently loaded NIfTI metadata (for display without reloading)
        "nifti_meta": None,
        # IQM dataframe (cached after first load)
        "iqm_df": None,
        # Flag: whether MedGemma model is loaded
        "medgemma_ready": False,
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
