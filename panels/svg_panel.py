"""
panels/svg_panel.py
-------------------
SVG pipeline-output viewer panel.

Renders SVG files produced by NiPreps pipelines (fMRIPrep, QSIPrep,
MRIQC) inside Streamlit iframes. Supports multi-file navigation,
zoom, and per-file annotations that are stored in session state.

This panel covers the MVP deliverable: "SVG panel" from the QC-Studio
project specification, which both competing applicants omitted entirely.
"""

from __future__ import annotations

import base64
from typing import List

import streamlit as st
import streamlit.components.v1 as components


# ── Heuristic label map: file-name fragments → human label ────────────────────
_SVG_LABEL_HINTS: dict[str, str] = {
    "brainmask":       "Brain mask overlay",
    "brain_mask":      "Brain mask overlay",
    "boldref":         "BOLD reference image",
    "bold_ref":        "BOLD reference image",
    "t1w":             "T1w structural",
    "t2w":             "T2w structural",
    "carpet":          "Carpet plot (confounds)",
    "confounds":       "Confound correlations",
    "motion":          "Head motion parameters",
    "surface":         "Cortical surface",
    "cortical":        "Cortical surface",
    "registration":    "Registration check",
    "coregist":        "Co-registration check",
    "fieldmap":        "Field-map",
    "susceptibility":  "Susceptibility distortion",
    "roi":             "ROI check",
    "anat":            "Anatomical",
    "func":            "Functional",
    "seg":             "Segmentation",
}


def _guess_label(file_name: str) -> str:
    """Return a human-readable label for an SVG based on its file name."""
    lower = file_name.lower()
    for fragment, label in _SVG_LABEL_HINTS.items():
        if fragment in lower:
            return label
    return file_name  # fallback: use raw name


def _svg_to_iframe_html(svg_bytes: bytes, zoom: float, bg_dark: bool) -> str:
    """Wrap SVG bytes in a scrollable HTML iframe for Streamlit.

    Parameters
    ----------
    svg_bytes:
        Raw SVG file bytes.
    zoom:
        CSS zoom level (1.0 = 100 %).
    bg_dark:
        If True, inject a dark-mode CSS filter (useful for white-bg SVGs).
    """
    b64 = base64.b64encode(svg_bytes).decode("utf-8")
    filter_css = "filter: invert(0.88) hue-rotate(180deg);" if bg_dark else ""

    return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    html, body {{
      width:100%; height:100%;
      background: {'#0a0e1a' if bg_dark else '#ffffff'};
      overflow:auto;
    }}
    img {{
      width: calc(100% * {zoom});
      display: block;
      {filter_css}
      image-rendering: crisp-edges;
    }}
  </style>
</head>
<body>
  <img src="data:image/svg+xml;base64,{b64}" alt="SVG pipeline output" />
</body>
</html>
"""


def render_svg_panel(svg_files: List) -> None:
    """Render the SVG outputs panel.

    Parameters
    ----------
    svg_files:
        List of Streamlit ``UploadedFile`` objects with SVG content.
    """
    st.markdown("### ◈ SVG Pipeline Outputs")
    st.caption(
        "Visualize SVG report panels from fMRIPrep, QSIPrep, MRIQC, and "
        "other NiPreps pipelines. Use the sidebar file uploader to add SVG files."
    )

    if not svg_files:
        st.info(
            "No SVG files uploaded. Upload `.svg` pipeline output files "
            "(e.g. fMRIPrep brainmask panels, carpet plots) using the sidebar."
        )

        # ── Demo hint ─────────────────────────────────────────────────────
        with st.expander("ℹ️  Where to find SVG pipeline outputs", expanded=False):
            st.markdown(
                """
**fMRIPrep** outputs SVG QC panels to:
```
derivatives/fmriprep/sub-<ID>/figures/*.svg
```

**MRIQC** outputs SVG IQM-distribution plots to:
```
derivatives/mriqc/sub-<ID>/figures/*.svg
```

**QSIPrep** outputs SVG tractography previews to:
```
derivatives/qsiprep/sub-<ID>/figures/*.svg
```

These SVG files can be dragged directly into the uploader above.
                """
            )
        return

    # ── Viewer controls ───────────────────────────────────────────────────
    col_nav, col_zoom, col_dark = st.columns([3, 1, 1])

    with col_nav:
        # Build display labels
        labels = [f"{i+1}. {_guess_label(f.name)}" for i, f in enumerate(svg_files)]
        selected_label = st.selectbox(
            "Select SVG panel", labels, key="svg_panel_select"
        )
        selected_idx = labels.index(selected_label)

    with col_zoom:
        zoom = st.select_slider(
            "Zoom", options=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
            value=1.0, key="svg_zoom"
        )

    with col_dark:
        dark_mode = st.checkbox(
            "Invert (dark)", value=True, key="svg_dark_mode",
            help="Apply dark-mode filter for white-background pipeline SVGs"
        )

    st.divider()

    # ── Main viewer ───────────────────────────────────────────────────────
    current_file = svg_files[selected_idx]
    svg_bytes = current_file.getvalue()

    viewer_height = st.slider(
        "Panel height (px)", 300, 900, 500, step=50, key="svg_height"
    )

    html_code = _svg_to_iframe_html(svg_bytes, zoom=zoom, bg_dark=dark_mode)
    components.html(html_code, height=viewer_height, scrolling=True)

    # ── File info ─────────────────────────────────────────────────────────
    st.caption(
        f"📄 `{current_file.name}` · {current_file.size / 1024:.1f} KB · "
        f"Panel {selected_idx + 1} of {len(svg_files)}"
    )

    # ── Per-file annotation ───────────────────────────────────────────────
    st.divider()
    st.markdown("#### Annotation")

    ann_key = f"svg_annotation_{current_file.name}"
    if ann_key not in st.session_state:
        st.session_state[ann_key] = ""

    annotation = st.text_area(
        f"Notes for `{current_file.name}`",
        value=st.session_state[ann_key],
        placeholder="e.g. Poor brain mask coverage — temporal lobe excluded",
        key=f"svg_ann_input_{selected_idx}",
        height=80,
    )

    col_save, col_clear = st.columns([1, 3])
    with col_save:
        if st.button("💾 Save note", key=f"svg_save_{selected_idx}"):
            st.session_state[ann_key] = annotation
            st.success("Annotation saved.")

    # ── Thumbnail strip ───────────────────────────────────────────────────
    if len(svg_files) > 1:
        st.divider()
        st.markdown("#### All SVG Panels")
        thumb_cols = st.columns(min(len(svg_files), 4))
        for i, (col, f) in enumerate(zip(thumb_cols, svg_files[:4])):
            with col:
                thumb_bytes = f.getvalue()
                thumb_html = _svg_to_iframe_html(
                    thumb_bytes, zoom=1.0, bg_dark=dark_mode
                )
                components.html(thumb_html, height=140, scrolling=False)
                st.caption(f"`{f.name[:22]}…`" if len(f.name) > 22 else f"`{f.name}`")

        if len(svg_files) > 4:
            st.caption(f"… and {len(svg_files) - 4} more panels. Use the dropdown to navigate.")
