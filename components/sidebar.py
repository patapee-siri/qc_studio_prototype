"""
components/sidebar.py
---------------------
Sidebar: file uploaders, subject ID input, and viewer controls.

Returns a dict of uploaded file objects consumed by each panel.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_sidebar() -> dict[str, Any]:
    """Render sidebar controls and return uploaded file references.

    Returns
    -------
    dict with keys:
        - ``mri``         : UploadedFile | None  — NIfTI (.nii / .nii.gz)
        - ``svgs``        : list[UploadedFile]   — SVG pipeline outputs
        - ``iqm``         : UploadedFile | None  — IQM TSV from MRIQC
        - ``subject_id``  : str                  — Subject identifier label
        - ``slice_files`` : list[UploadedFile]   — 2D slices (PNG/JPG/SVG) for 3D reconstruction
    """
    with st.sidebar:
        # ── Branding ─────────────────────────────────────────────────────
        st.markdown(
            "<p style='font-family:JetBrains Mono,monospace;"
            "font-size:0.68rem;color:#2a4a72;letter-spacing:2px;"
            "text-transform:uppercase;margin-bottom:16px;'>"
            "◈ QC-Studio · Nipoppy</p>",
            unsafe_allow_html=True,
        )

        # ── Subject ID ────────────────────────────────────────────────────
        st.markdown("**Subject**")
        subject_id: str = st.text_input(
            "Subject ID",
            value="sub-001",
            label_visibility="collapsed",
            placeholder="e.g. sub-001",
        )

        st.divider()

        # ── MRI upload ────────────────────────────────────────────────────
        st.markdown("**MRI Volume**")
        mri_file = st.file_uploader(
            "Upload NIfTI",
            type=["nii", "gz"],
            label_visibility="collapsed",
            help="Accepts .nii and .nii.gz (NIfTI-1 and NIfTI-2)",
        )
        if mri_file:
            st.caption(f"📦 `{mri_file.name}`  ·  {mri_file.size / 1024:.1f} KB")

        st.divider()

        # ── SVG uploads ───────────────────────────────────────────────────
        st.markdown("**SVG Pipeline Outputs**")
        svg_files = st.file_uploader(
            "Upload SVGs",
            type=["svg"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help="fMRIPrep / QSIPrep / MRIQC SVG report panels",
        )
        if svg_files:
            st.caption(f"📂 {len(svg_files)} SVG file(s) loaded")

        st.divider()

        # ── 2D slice uploads ──────────────────────────────────────────────
        st.markdown("**2D Slices (PNG / JPG / SVG)**")
        slice_files = st.file_uploader(
            "Upload 2D slices",
            type=["png", "jpg", "jpeg", "svg"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            help=(
                "Upload multiple 2D images (one per slice) to reconstruct "
                "a 3D NIfTI volume. Stacking controls appear in the NiiVue panel."
            ),
        )
        if slice_files:
            st.caption(f"🖼 {len(slice_files)} slice file(s) loaded")

        st.divider()

        # ── IQM TSV ───────────────────────────────────────────────────────
        st.markdown("**Image Quality Metrics (TSV)**")
        iqm_file = st.file_uploader(
            "Upload IQM TSV",
            type=["tsv", "csv"],
            label_visibility="collapsed",
            help="group_T1w.tsv / group_bold.tsv from MRIQC",
        )
        if iqm_file:
            st.caption(f"📊 `{iqm_file.name}`")

        st.divider()

        # ── Pipeline info ─────────────────────────────────────────────────
        st.markdown(
            "<p style='font-family:JetBrains Mono,monospace;"
            "font-size:0.68rem;color:#2a4a72;margin-top:8px;'>"
            "Compatible pipelines:<br/>"
            "fMRIPrep · QSIPrep<br/>"
            "MRIQC · FreeSurfer<br/>"
            "Nipoppy framework</p>",
            unsafe_allow_html=True,
        )

    return {
        "mri": mri_file,
        "svgs": svg_files or [],
        "iqm": iqm_file,
        "subject_id": subject_id,
        "slice_files": slice_files or [],
    }
