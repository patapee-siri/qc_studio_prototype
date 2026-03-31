"""
QC-Studio: Integrated Image Quality Control Toolkit for MRI Datasets
=====================================================================
A production-ready Streamlit application for semi-automated neuroimaging
quality control within the Nipoppy neuroinformatics framework.

Supports: NiiVue multiplanar viewing, SVG pipeline outputs, IQM dashboards,
          MedGemma-powered annotations, and QC decision export.
"""

import streamlit as st

# ── Page config MUST be first Streamlit call ─────────────────────────────────
st.set_page_config(
    page_title="QC-Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: dark clinical aesthetic ──────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=Sora:wght@300;400;600;700&display=swap');

    /* ── Global reset ── */
    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }
    .stApp {
        background: #0a0e1a;
        color: #c8d6e5;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0d1221 !important;
        border-right: 1px solid #1e2d45;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p {
        color: #8fa3bc !important;
        font-size: 0.82rem;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Header brand ── */
    .qcs-header {
        display: flex;
        align-items: baseline;
        gap: 12px;
        padding: 6px 0 18px 0;
        border-bottom: 1px solid #1e2d45;
        margin-bottom: 20px;
    }
    .qcs-header h1 {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 1.7rem;
        color: #e0ecff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .qcs-header .badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        padding: 3px 8px;
        border-radius: 3px;
        background: #1a2d4a;
        color: #4d9de0;
        border: 1px solid #2a4a72;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #0d1221;
        border-bottom: 1px solid #1e2d45;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #5a7a99;
        padding: 10px 22px;
        border-bottom: 2px solid transparent;
        letter-spacing: 0.5px;
    }
    .stTabs [aria-selected="true"] {
        color: #4d9de0 !important;
        border-bottom-color: #4d9de0 !important;
        background: transparent !important;
    }

    /* ── Cards / metric boxes ── */
    .qcs-card {
        background: #0d1629;
        border: 1px solid #1e2d45;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .qcs-metric {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.4rem;
        font-weight: 500;
        color: #e0ecff;
    }
    .qcs-label {
        font-size: 0.72rem;
        color: #5a7a99;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 2px;
    }

    /* ── Status badges ── */
    .badge-pass {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 3px;
        background: #0d2a1a;
        color: #3ddc84;
        border: 1px solid #1e5c38;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 1px;
    }
    .badge-fail {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 3px;
        background: #2a0d0d;
        color: #ff6b6b;
        border: 1px solid #5c1e1e;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 1px;
    }
    .badge-warn {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 3px;
        background: #2a1e0d;
        color: #ffa726;
        border: 1px solid #5c3e1e;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 1px;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: #1a2d4a !important;
        color: #4d9de0 !important;
        border: 1px solid #2a4a72 !important;
        border-radius: 4px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.5px !important;
        padding: 6px 16px !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button:hover {
        background: #243d5e !important;
        border-color: #4d9de0 !important;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        border: 1px dashed #2a4a72 !important;
        border-radius: 6px !important;
        background: #0d1629 !important;
    }

    /* ── Divider ── */
    hr {
        border-color: #1e2d45 !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #2a4a72; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Imports (after page config) ───────────────────────────────────────────────
from components.sidebar import render_sidebar
from panels.niivue_panel import render_niivue_panel
from panels.svg_panel import render_svg_panel
from panels.iqm_panel import render_iqm_panel
from panels.medgemma_panel import render_medgemma_panel
from utils.session import init_session_state
from utils.export import render_export_section

# ── Session state ─────────────────────────────────────────────────────────────
init_session_state()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="qcs-header">
        <h1>🧠 QC-Studio</h1>
        <span class="badge">Nipoppy · v0.1.0-gsoc</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar: file upload + controls ───────────────────────────────────────────
uploaded = render_sidebar()

mri_file   = uploaded.get("mri")
svg_files  = uploaded.get("svgs", [])
iqm_file   = uploaded.get("iqm")
subject_id = uploaded.get("subject_id", "sub-unknown")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_viewer, tab_svg, tab_iqm, tab_ai, tab_export = st.tabs(
    ["🧠 NiiVue Viewer", "◈ SVG Outputs", "📊 IQM Dashboard", "🤖 AI Radiologist", "↓ Export"]
)

with tab_viewer:
    render_niivue_panel(mri_file)

with tab_svg:
    render_svg_panel(svg_files)

with tab_iqm:
    render_iqm_panel(iqm_file, subject_id)

with tab_ai:
    render_medgemma_panel(mri_file, iqm_file)

with tab_export:
    render_export_section(subject_id)
