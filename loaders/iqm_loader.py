"""
loaders/iqm_loader.py
---------------------
Load and validate MRIQC Image Quality Metrics (IQM) from TSV / CSV.

MRIQC produces ``group_T1w.tsv`` and ``group_bold.tsv`` files with one
row per subject-session-run. This loader parses those files, identifies
known IQM columns, and attaches reference-range metadata used by the
IQM dashboard panel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import streamlit as st


# ── MRIQC reference ranges ────────────────────────────────────────────────────
# Based on: Esteban et al. (2017) MRIQC paper + community norms.
# Direction: "higher_better" or "lower_better"
IQM_REFERENCE_RANGES: dict[str, dict] = {
    # Structural T1w metrics
    "snr":         {"min": 5.0,  "max": 40.0, "good_min": 15.0, "direction": "higher_better", "unit": "",      "label": "SNR"},
    "snr_gm":      {"min": 3.0,  "max": 35.0, "good_min": 12.0, "direction": "higher_better", "unit": "",      "label": "SNR (GM)"},
    "snr_wm":      {"min": 3.0,  "max": 35.0, "good_min": 12.0, "direction": "higher_better", "unit": "",      "label": "SNR (WM)"},
    "snr_csf":     {"min": 1.0,  "max": 20.0, "good_min":  5.0, "direction": "higher_better", "unit": "",      "label": "SNR (CSF)"},
    "cnr":         {"min": 0.5,  "max": 8.0,  "good_min":  2.0, "direction": "higher_better", "unit": "",      "label": "CNR"},
    "efc":         {"min": 0.3,  "max": 0.9,  "good_max":  0.6, "direction": "lower_better",  "unit": "",      "label": "EFC"},
    "fber":        {"min": 50.0, "max": 2000, "good_min": 500.0,"direction": "higher_better", "unit": "",      "label": "FBER"},
    "wm2max":      {"min": 0.0,  "max": 1.0,  "good_min":  0.6, "direction": "higher_better", "unit": "",      "label": "WM2MAX"},
    "qi_1":        {"min": 0.0,  "max": 0.2,  "good_max":  0.05,"direction": "lower_better",  "unit": "",      "label": "QI1"},
    "cjv":         {"min": 0.1,  "max": 1.5,  "good_max":  0.5, "direction": "lower_better",  "unit": "",      "label": "CJV"},
    "icvs_gm":     {"min": 0.3,  "max": 0.6,  "direction": "higher_better", "unit": "",        "label": "ICV-GM"},
    "icvs_wm":     {"min": 0.2,  "max": 0.5,  "direction": "higher_better", "unit": "",        "label": "ICV-WM"},
    # Functional BOLD metrics
    "tsnr":        {"min": 20.0, "max": 120.0,"good_min": 40.0, "direction": "higher_better", "unit": "",      "label": "tSNR"},
    "dvars_std":   {"min": 0.0,  "max": 2.0,  "good_max":  1.2, "direction": "lower_better",  "unit": "",      "label": "DVARS (std)"},
    "gcor":        {"min": -0.2, "max": 0.5,  "good_max":  0.1, "direction": "lower_better",  "unit": "",      "label": "GCOR"},
    "gsr_x":       {"min": -0.3, "max": 0.3,  "good_max":  0.1, "direction": "lower_better",  "unit": "",      "label": "GSR (x)"},
    "gsr_y":       {"min": -0.3, "max": 0.3,  "good_max":  0.1, "direction": "lower_better",  "unit": "",      "label": "GSR (y)"},
    "fd_mean":     {"min": 0.0,  "max": 2.0,  "good_max":  0.5, "direction": "lower_better",  "unit": "mm",    "label": "FD (mean)"},
    "fd_perc":     {"min": 0.0,  "max": 100,  "good_max": 20.0, "direction": "lower_better",  "unit": "%",     "label": "FD >0.5mm (%)"},
    "aor":         {"min": 0.0,  "max": 0.5,  "good_max":  0.1, "direction": "lower_better",  "unit": "",      "label": "AOR"},
    "aqi":         {"min": 0.0,  "max": 1.0,  "good_max":  0.3, "direction": "lower_better",  "unit": "",      "label": "AQI"},
    "fwhm_avg":    {"min": 1.5,  "max": 6.0,  "direction": "lower_better",  "unit": "mm",      "label": "FWHM (avg)"},
}


@dataclass
class IQMDataset:
    """Parsed IQM dataset ready for visualization."""

    df: pd.DataFrame
    """Full DataFrame (all columns)."""

    iqm_columns: list[str]
    """Column names recognised as MRIQC IQMs."""

    has_subject_col: bool = False
    """True if a 'subject_id' or 'bids_name' column is present."""

    modality: str = "unknown"
    """Detected modality: 'T1w', 'BOLD', or 'unknown'."""

    warnings: list[str] = field(default_factory=list)
    """Non-fatal parse warnings."""


@st.cache_data(show_spinner=False)
def load_iqm(file_bytes: bytes, file_name: str) -> IQMDataset:
    """Parse an MRIQC group TSV/CSV into an IQMDataset.

    Parameters
    ----------
    file_bytes:
        Raw bytes from ``uploaded_file.getvalue()``.
    file_name:
        Original file name (used to infer separator and modality).

    Returns
    -------
    IQMDataset

    Raises
    ------
    ValueError
        If the file cannot be parsed as tabular data.
    """
    warnings: list[str] = []

    # ── Choose separator ──────────────────────────────────────────────────
    sep = "\t" if file_name.lower().endswith(".tsv") else ","

    try:
        import io
        df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
    except Exception as exc:
        raise ValueError(f"Cannot parse '{file_name}' as tabular data: {exc}") from exc

    if df.empty:
        raise ValueError(f"'{file_name}' is empty.")

    # ── Normalise column names to lowercase ───────────────────────────────
    df.columns = [c.strip().lower() for c in df.columns]

    # ── Detect subject column ─────────────────────────────────────────────
    subject_col_candidates = ["subject_id", "bids_name", "participant_id", "sub"]
    has_subject_col = any(c in df.columns for c in subject_col_candidates)

    # ── Detect modality ───────────────────────────────────────────────────
    modality = "unknown"
    if "t1w" in file_name.lower():
        modality = "T1w"
    elif "bold" in file_name.lower() or "func" in file_name.lower():
        modality = "BOLD"
    elif "tsnr" in df.columns or "dvars_std" in df.columns:
        modality = "BOLD"
    elif "snr_gm" in df.columns or "cjv" in df.columns:
        modality = "T1w"

    # ── Identify known IQM columns ────────────────────────────────────────
    known_keys = set(IQM_REFERENCE_RANGES.keys())
    iqm_columns = [c for c in df.columns if c in known_keys]

    if not iqm_columns:
        # Fallback: accept any numeric column that isn't subject-like
        skip = set(subject_col_candidates + ["session_id", "run_id", "task"])
        iqm_columns = [
            c for c in df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(df[c])
        ]
        warnings.append(
            "No standard MRIQC IQM column names detected. "
            "Showing all numeric columns."
        )

    # ── Coerce IQM columns to float ───────────────────────────────────────
    for col in iqm_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return IQMDataset(
        df=df,
        iqm_columns=iqm_columns,
        has_subject_col=has_subject_col,
        modality=modality,
        warnings=warnings,
    )


def qc_flag_single_row(row: pd.Series) -> str:
    """Heuristic QC flag for a single IQM row.

    Returns ``'PASS'``, ``'FAIL'``, or ``'WARN'``.
    """
    fail_conditions = []
    warn_conditions = []

    checks = {
        "snr":      ("higher_better", 8.0,  12.0),
        "cnr":      ("higher_better", 1.0,   2.0),
        "efc":      ("lower_better",  0.7,   0.65),
        "cjv":      ("lower_better",  0.8,   0.6),
        "fd_mean":  ("lower_better",  1.0,   0.5),
        "fd_perc":  ("lower_better",  30.0,  20.0),
        "tsnr":     ("higher_better", 25.0,  40.0),
        "dvars_std":("lower_better",  1.8,   1.2),
    }

    for metric, (direction, fail_thresh, warn_thresh) in checks.items():
        if metric not in row.index:
            continue
        val = row[metric]
        if pd.isna(val):
            continue
        if direction == "higher_better":
            if val < fail_thresh:
                fail_conditions.append(metric)
            elif val < warn_thresh:
                warn_conditions.append(metric)
        else:  # lower_better
            if val > fail_thresh:
                fail_conditions.append(metric)
            elif val > warn_thresh:
                warn_conditions.append(metric)

    if fail_conditions:
        return "FAIL"
    if warn_conditions:
        return "WARN"
    return "PASS"
